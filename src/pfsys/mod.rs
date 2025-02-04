/// EVM related proving and verification
pub mod evm;

use crate::circuit::CheckMode;
use crate::commands::{data_path, Cli, RunArgs};
use crate::execute::ExecutionError;
use crate::fieldutils::i128_to_felt;
use crate::graph::{utilities::vector_to_quantized, Model, ModelCircuit};
use crate::tensor::ops::pack;
use crate::tensor::{Tensor, TensorType};
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::circuit::Value;
use halo2_proofs::dev::MockProver;
use halo2_proofs::plonk::{
    create_proof, keygen_pk, keygen_vk, verify_proof, Circuit, ProvingKey, VerifyingKey,
};
use halo2_proofs::poly::commitment::{CommitmentScheme, Params, ParamsProver, Prover, Verifier};
use halo2_proofs::poly::VerificationStrategy;
use halo2_proofs::transcript::{EncodedChallenge, TranscriptReadBuffer, TranscriptWriterBuffer};
use halo2curves::group::ff::PrimeField;
use halo2curves::serde::SerdeObject;
use halo2curves::CurveAffine;
use log::{debug, info, trace};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use snark_verifier::system::halo2::{compile, Config};
use snark_verifier::verifier::plonk::PlonkProtocol;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Cursor, Read, Write};
use std::marker::PhantomData;
use std::ops::Deref;
use std::path::PathBuf;
use std::time::Instant;
use thiserror::Error as thisError;

#[derive(thisError, Debug)]
/// Errors related to pfsys
pub enum PfSysError {
    /// Packing exponent is too large
    #[error("largest packing exponent exceeds max. try reducing the scale")]
    PackingExponent,
}

/// The input tensor data and shape, and output data for the computational graph (model) as floats.
/// For example, the input might be the image data for a neural network, and the output class scores.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelInput {
    /// Inputs to the model / computational graph.
    pub input_data: Vec<Vec<f32>>,
    /// The shape of said inputs.
    pub input_shapes: Vec<Vec<usize>>,
    /// The expected output of the model (can be empty vectors if outputs are not being constrained).
    pub output_data: Vec<Vec<f32>>,
}

/// Defines the proof generated by a model / circuit suitably for serialization/deserialization.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Snarkbytes {
    num_instance: Vec<usize>,
    /// Public inputs to the model.
    pub instances: Vec<Vec<Vec<u8>>>,
    /// The generated proof, as a vector of bytes.
    pub proof: Vec<u8>,
}

/// An application snark with proof and instance variables ready for aggregation (raw field element)
#[derive(Debug, Clone)]
pub struct Snark<F: FieldExt + SerdeObject, C: CurveAffine> {
    protocol: Option<PlonkProtocol<C>>,
    /// public instances of the snark
    pub instances: Vec<Vec<F>>,
    /// the proof
    pub proof: Vec<u8>,
}

impl<F: FieldExt + SerdeObject, C: CurveAffine> Snark<F, C> {
    /// Create a new application snark from proof and instance variables ready for aggregation
    pub fn new(protocol: PlonkProtocol<C>, instances: Vec<Vec<F>>, proof: Vec<u8>) -> Self {
        Self {
            protocol: Some(protocol),
            instances,
            proof,
        }
    }

    /// Saves the Proof to a specified `proof_path`.
    pub fn save(&self, proof_path: &PathBuf) -> Result<(), Box<dyn Error>> {
        let self_i128 = Snarkbytes {
            num_instance: self.protocol.as_ref().unwrap().num_instance.clone(),
            instances: self
                .instances
                .iter()
                .map(|i| i.iter().map(|e| e.to_raw_bytes()).collect::<Vec<Vec<u8>>>())
                .collect::<Vec<Vec<Vec<u8>>>>(),
            proof: self.proof.clone(),
        };

        let serialized = serde_json::to_string(&self_i128).map_err(Box::<dyn Error>::from)?;

        let mut file = std::fs::File::create(proof_path).map_err(Box::<dyn Error>::from)?;
        file.write_all(serialized.as_bytes())
            .map_err(Box::<dyn Error>::from)
    }

    /// Load a json serialized proof from the provided path.
    pub fn load<Scheme: CommitmentScheme<Curve = C, Scalar = F>>(
        proof_path: &PathBuf,
        params: Option<&Scheme::ParamsProver>,
        vk: Option<&VerifyingKey<C>>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut file = File::open(proof_path).map_err(Box::<dyn Error>::from)?;
        let mut data = String::new();
        file.read_to_string(&mut data)
            .map_err(Box::<dyn Error>::from)?;
        let snark_bytes: Snarkbytes =
            serde_json::from_str(&data).map_err(Box::<dyn Error>::from)?;

        let instances = snark_bytes
            .instances
            .iter()
            .map(|i| {
                i.iter()
                    .map(|e| Scheme::Scalar::from_raw_bytes_unchecked(e))
                    .collect::<Vec<Scheme::Scalar>>()
            })
            .collect::<Vec<Vec<Scheme::Scalar>>>();

        trace!("instances {:?}", instances);

        if params.is_none() || vk.is_none() {
            Ok(Snark {
                protocol: None,
                instances,
                proof: snark_bytes.proof,
            })
        } else {
            let protocol = compile(
                params.unwrap(),
                vk.unwrap(),
                Config::kzg().with_num_instance(snark_bytes.num_instance.clone()),
            );

            Ok(Snark {
                protocol: Some(protocol),
                instances,
                proof: snark_bytes.proof,
            })
        }
    }
}

/// An application snark with proof and instance variables ready for aggregation (wrapped field element)
#[derive(Clone, Debug)]
pub struct SnarkWitness<F: FieldExt, C: CurveAffine> {
    protocol: Option<PlonkProtocol<C>>,
    instances: Vec<Vec<Value<F>>>,
    proof: Value<Vec<u8>>,
}

impl<F: FieldExt, C: CurveAffine> SnarkWitness<F, C> {
    fn without_witnesses(&self) -> Self {
        SnarkWitness {
            protocol: self.protocol.clone(),
            instances: self
                .instances
                .iter()
                .map(|instances| vec![Value::unknown(); instances.len()])
                .collect(),
            proof: Value::unknown(),
        }
    }

    fn proof(&self) -> Value<&[u8]> {
        self.proof.as_ref().map(Vec::as_slice)
    }
}

impl<F: FieldExt + SerdeObject, C: CurveAffine> From<Snark<F, C>> for SnarkWitness<F, C> {
    fn from(snark: Snark<F, C>) -> Self {
        Self {
            protocol: snark.protocol,
            instances: snark
                .instances
                .into_iter()
                .map(|instances| instances.into_iter().map(Value::known).collect())
                .collect(),
            proof: Value::known(snark.proof),
        }
    }
}

type CircuitInputs<F> = (ModelCircuit<F>, Vec<Vec<F>>);

/// Initialize the model circuit and quantize the provided float inputs from the provided `ModelInput`.
pub fn prepare_model_circuit_and_public_input<F: FieldExt + TensorType>(
    data: &ModelInput,
    cli: &Cli,
) -> Result<CircuitInputs<F>, Box<dyn Error>> {
    let model = Model::from_ezkl_conf(cli.clone())?;
    let out_scales = model.get_output_scales();
    let circuit = prepare_model_circuit(data, &cli.args)?;

    // quantize the supplied data using the provided scale.
    // the ordering here is important, we want the inputs to come before the outputs
    // as they are configured in that order as Column<Instances>
    let mut public_inputs = vec![];
    if model.visibility.input.is_public() {
        for v in data.input_data.iter() {
            let t = vector_to_quantized(v, &Vec::from([v.len()]), 0.0, model.run_args.scale)?;
            public_inputs.push(t);
        }
    }
    if model.visibility.output.is_public() {
        for (idx, v) in data.output_data.iter().enumerate() {
            let mut t = vector_to_quantized(v, &Vec::from([v.len()]), 0.0, out_scales[idx])?;
            let len = t.len();
            if cli.args.pack_base > 1 {
                let max_exponent = (((len - 1) as u32) * (cli.args.scale + 1)) as f64;
                if max_exponent > (i128::MAX as f64).log(cli.args.pack_base as f64) {
                    return Err(Box::new(PfSysError::PackingExponent));
                }
                t = pack(&t, cli.args.pack_base as i128, cli.args.scale)?;
            }
            public_inputs.push(t);
        }
    }
    info!(
        "public inputs lengths: {:?}",
        public_inputs
            .iter()
            .map(|i| i.len())
            .collect::<Vec<usize>>()
    );
    trace!("{:?}", public_inputs);

    let pi_inner: Vec<Vec<F>> = public_inputs
        .iter()
        .map(|i| i.iter().map(|e| i128_to_felt::<F>(*e)).collect::<Vec<F>>())
        .collect::<Vec<Vec<F>>>();

    Ok((circuit, pi_inner))
}

/// Initialize the model circuit
pub fn prepare_model_circuit<F: FieldExt>(
    data: &ModelInput,
    args: &RunArgs,
) -> Result<ModelCircuit<F>, Box<dyn Error>> {
    // quantize the supplied data using the provided scale.
    let mut inputs: Vec<Tensor<i128>> = vec![];
    for (input, shape) in data.input_data.iter().zip(data.input_shapes.clone()) {
        let t = vector_to_quantized(input, &shape, 0.0, args.scale)?;
        inputs.push(t);
    }

    Ok(ModelCircuit::<F> {
        inputs,
        _marker: PhantomData,
    })
}

/// Deserializes the required inputs to a model at path `datapath` to a [ModelInput] struct.
pub fn prepare_data(datapath: String) -> Result<ModelInput, Box<dyn Error>> {
    let mut file = File::open(data_path(datapath)).map_err(Box::<dyn Error>::from)?;
    let mut data = String::new();
    file.read_to_string(&mut data)
        .map_err(Box::<dyn Error>::from)?;
    serde_json::from_str(&data).map_err(Box::<dyn Error>::from)
}

/// Helper function for generating SRS. !!! Only use for testing
pub fn gen_srs<Scheme: CommitmentScheme>(k: u32) -> Scheme::ParamsProver {
    Scheme::ParamsProver::new(k)
}

/// Creates a [VerifyingKey] and [ProvingKey] for a [ModelCircuit] (`circuit`) with specific [CommitmentScheme] parameters (`params`).
pub fn create_keys<Scheme: CommitmentScheme, F: FieldExt + TensorType, C: Circuit<F>>(
    circuit: &C,
    params: &'_ Scheme::ParamsProver,
) -> Result<ProvingKey<Scheme::Curve>, halo2_proofs::plonk::Error>
where
    C: Circuit<Scheme::Scalar>,
{
    //	Real proof
    let empty_circuit = <C as Circuit<F>>::without_witnesses(circuit);

    // Initialize the proving key
    let now = Instant::now();
    trace!("preparing VK");
    let vk = keygen_vk(params, &empty_circuit)?;
    info!("VK took {}", now.elapsed().as_secs());
    let now = Instant::now();
    let pk = keygen_pk(params, vk, &empty_circuit)?;
    info!("PK took {}", now.elapsed().as_secs());
    Ok(pk)
}

/// a wrapper around halo2's create_proof
pub fn create_proof_circuit<
    'params,
    Scheme: CommitmentScheme,
    F: FieldExt + TensorType,
    C: Circuit<F>,
    P: Prover<'params, Scheme>,
    V: Verifier<'params, Scheme>,
    Strategy: VerificationStrategy<'params, Scheme, V>,
    E: EncodedChallenge<Scheme::Curve>,
    TW: TranscriptWriterBuffer<Vec<u8>, Scheme::Curve, E>,
    TR: TranscriptReadBuffer<Cursor<Vec<u8>>, Scheme::Curve, E>,
>(
    circuit: C,
    instances: Vec<Vec<Scheme::Scalar>>,
    params: &'params Scheme::ParamsProver,
    pk: &ProvingKey<Scheme::Curve>,
    strategy: Strategy,
    check_mode: CheckMode,
) -> Result<Snark<Scheme::Scalar, Scheme::Curve>, Box<dyn Error>>
where
    C: Circuit<Scheme::Scalar>,
    Scheme::ParamsVerifier: 'params,
    Scheme::Scalar: SerdeObject,
{
    // quickly mock prove as a sanity check
    if check_mode == CheckMode::SAFE {
        debug!("running mock prover");
        let prover = MockProver::run(params.k(), &circuit, instances.clone())
            .map_err(Box::<dyn Error>::from)?;
        prover
            .verify()
            .map_err(|e| Box::<dyn Error>::from(ExecutionError::VerifyError(e)))?;
    }

    let mut transcript = TranscriptWriterBuffer::<_, Scheme::Curve, _>::init(vec![]);
    let mut rng = OsRng;
    let number_instance = instances.iter().map(|x| x.len()).collect();
    trace!("number_instance {:?}", number_instance);
    let protocol = compile(
        params,
        pk.get_vk(),
        Config::kzg().with_num_instance(number_instance),
    );

    let pi_inner = instances
        .iter()
        .map(|e| e.deref())
        .collect::<Vec<&[Scheme::Scalar]>>();
    let pi_inner: &[&[&[Scheme::Scalar]]] = &[&pi_inner];
    trace!("instances {:?}", instances);

    let now = Instant::now();
    create_proof::<Scheme, P, _, _, TW, _>(
        params,
        pk,
        &[circuit],
        pi_inner,
        &mut rng,
        &mut transcript,
    )?;
    let proof = transcript.finalize();
    info!("Proof took {}", now.elapsed().as_secs());

    let checkable_pf = Snark::new(protocol, instances, proof);

    // sanity check that the generated proof is valid
    if check_mode == CheckMode::SAFE {
        debug!("verifying generated proof");
        let verifier_params = params.verifier_params();
        verify_proof_circuit::<F, V, Scheme, Strategy, E, TR>(
            &checkable_pf,
            verifier_params,
            pk.get_vk(),
            strategy,
        )?;
    }

    Ok(checkable_pf)
}

/// A wrapper around halo2's verify_proof
pub fn verify_proof_circuit<
    'params,
    F: FieldExt,
    V: Verifier<'params, Scheme>,
    Scheme: CommitmentScheme,
    Strategy: VerificationStrategy<'params, Scheme, V>,
    E: EncodedChallenge<Scheme::Curve>,
    TR: TranscriptReadBuffer<Cursor<Vec<u8>>, Scheme::Curve, E>,
>(
    snark: &Snark<Scheme::Scalar, Scheme::Curve>,
    params: &'params Scheme::ParamsVerifier,
    vk: &VerifyingKey<Scheme::Curve>,
    strategy: Strategy,
) -> Result<Strategy::Output, halo2_proofs::plonk::Error>
where
    Scheme::Scalar: SerdeObject,
{
    let pi_inner = snark
        .instances
        .iter()
        .map(|e| e.deref())
        .collect::<Vec<&[Scheme::Scalar]>>();
    let instances: &[&[&[Scheme::Scalar]]] = &[&pi_inner];
    trace!("instances {:?}", instances);

    let now = Instant::now();
    let mut transcript = TranscriptReadBuffer::init(Cursor::new(snark.proof.clone()));
    info!("verify took {}", now.elapsed().as_secs());
    verify_proof::<Scheme, V, _, TR, _>(params, vk, strategy, instances, &mut transcript)
}

/// Loads a [VerifyingKey] at `path`.
pub fn load_vk<Scheme: CommitmentScheme, F: FieldExt + TensorType, C: Circuit<F>>(
    path: PathBuf,
) -> Result<VerifyingKey<Scheme::Curve>, Box<dyn Error>>
where
    C: Circuit<Scheme::Scalar>,
    Scheme::Curve: SerdeObject + CurveAffine,
    Scheme::Scalar: PrimeField + SerdeObject,
{
    info!("loading verification key from {:?}", path);
    let f = File::open(path).map_err(Box::<dyn Error>::from)?;
    let mut reader = BufReader::new(f);
    VerifyingKey::<Scheme::Curve>::read::<_, C>(&mut reader, halo2_proofs::SerdeFormat::RawBytes)
        .map_err(Box::<dyn Error>::from)
}

/// Loads a [ProvingKey] at `path`.
pub fn load_pk<Scheme: CommitmentScheme, F: FieldExt + TensorType, C: Circuit<F>>(
    path: PathBuf,
) -> Result<ProvingKey<Scheme::Curve>, Box<dyn Error>>
where
    C: Circuit<Scheme::Scalar>,
    Scheme::Curve: SerdeObject + CurveAffine,
    Scheme::Scalar: PrimeField + SerdeObject,
{
    info!("loading proving key from {:?}", path);
    let f = File::open(path).map_err(Box::<dyn Error>::from)?;
    let mut reader = BufReader::new(f);
    ProvingKey::<Scheme::Curve>::read::<_, C>(&mut reader, halo2_proofs::SerdeFormat::RawBytes)
        .map_err(Box::<dyn Error>::from)
}

/// Loads the [CommitmentScheme::ParamsVerifier] at `path`.
pub fn load_params<Scheme: CommitmentScheme>(
    path: PathBuf,
) -> Result<Scheme::ParamsVerifier, Box<dyn Error>> {
    info!("loading params from {:?}", path);
    let f = File::open(path).map_err(Box::<dyn Error>::from)?;
    let mut reader = BufReader::new(f);
    Params::<'_, Scheme::Curve>::read(&mut reader).map_err(Box::<dyn Error>::from)
}

/// Saves a [ProvingKey] to `path`.
pub fn save_pk<Scheme: CommitmentScheme>(
    path: &PathBuf,
    vk: &ProvingKey<Scheme::Curve>,
) -> Result<(), io::Error>
where
    Scheme::Curve: SerdeObject + CurveAffine,
    Scheme::Scalar: PrimeField + SerdeObject,
{
    info!("saving proving key 💾");
    let f = File::create(path)?;
    let mut writer = BufWriter::new(f);
    vk.write(&mut writer, halo2_proofs::SerdeFormat::RawBytes)?;
    writer.flush()?;
    Ok(())
}

/// Saves a [VerifyingKey] to `path`.
pub fn save_vk<Scheme: CommitmentScheme>(
    path: &PathBuf,
    vk: &VerifyingKey<Scheme::Curve>,
) -> Result<(), io::Error>
where
    Scheme::Curve: SerdeObject + CurveAffine,
    Scheme::Scalar: PrimeField + SerdeObject,
{
    info!("saving verification key 💾");
    let f = File::create(path)?;
    let mut writer = BufWriter::new(f);
    vk.write(&mut writer, halo2_proofs::SerdeFormat::RawBytes)?;
    writer.flush()?;
    Ok(())
}

/// Saves [CommitmentScheme] parameters to `path`.
pub fn save_params<Scheme: CommitmentScheme>(
    path: &PathBuf,
    params: &'_ Scheme::ParamsVerifier,
) -> Result<(), io::Error> {
    info!("saving parameters 💾");
    let f = File::create(path)?;
    let mut writer = BufWriter::new(f);
    params.write(&mut writer)?;
    writer.flush()?;
    Ok(())
}

////////////////////////

#[cfg(test)]
mod tests {
    use std::io::copy;

    use super::*;
    use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
    use halo2curves::bn256::Bn256;
    use tempfile::Builder;

    #[tokio::test]
    async fn test_can_load_pre_generated_srs() {
        let tmp_dir = Builder::new().prefix("example").tempdir().unwrap();
        // lets hope this link never rots
        let target = "https://trusted-setup-halo2kzg.s3.eu-central-1.amazonaws.com/hermez-raw-1";
        let response = reqwest::get(target).await.unwrap();

        let fname = response
            .url()
            .path_segments()
            .and_then(|segments| segments.last())
            .and_then(|name| if name.is_empty() { None } else { Some(name) })
            .unwrap_or("tmp.bin");

        info!("file to download: '{}'", fname);
        let fname = tmp_dir.path().join(fname);
        info!("will be located under: '{:?}'", fname);
        let mut dest = File::create(fname.clone()).unwrap();
        let content = response.bytes().await.unwrap();
        copy(&mut &content[..], &mut dest).unwrap();
        let res = load_params::<KZGCommitmentScheme<Bn256>>(fname);
        assert!(res.is_ok())
    }

    #[tokio::test]
    async fn test_can_load_saved_srs() {
        let tmp_dir = Builder::new().prefix("example").tempdir().unwrap();
        let fname = tmp_dir.path().join("kzg.params");
        let srs = gen_srs::<KZGCommitmentScheme<Bn256>>(1);
        let res = save_params::<KZGCommitmentScheme<Bn256>>(&fname, &srs);
        assert!(res.is_ok());
        let res = load_params::<KZGCommitmentScheme<Bn256>>(fname);
        assert!(res.is_ok())
    }
}
