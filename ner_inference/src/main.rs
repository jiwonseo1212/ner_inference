#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
mod bert_classification;
// use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};
use bert_classification::{BertModelWithClassifier, Config, HiddenAct, DTYPE};
use candle_transformers::models::with_tracing;
use anyhow::{Error as E, Result};
use candle_core::Tensor;
use candle_nn::VarBuilder;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Use the pytorch weights rather than the safetensors ones
    #[arg(long)]
    use_pth: bool,

    /// The number of times to run the prompt.
    #[arg(long, default_value = "1")]
    n: usize,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,

    /// Use tanh based approximation for Gelu instead of erf implementation.
    #[arg(long, default_value = "false")]
    approximate_gelu: bool,
}

impl Args {

    fn build_model_and_tokenizer_config(&self) -> Result<(BertModelWithClassifier, Tokenizer, Config)> {
        let device = candle_examples::device(self.cpu)?;
        let default_model = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let default_revision = "refs/pr/21".to_string();


        let (model_id, revision) = match (self.model_id.to_owned(), self.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };
        let model_path = Path::new(& model_id);
        let (config_filename, tokenizer_filename, weights_filename)  = if model_path.exists() {
            let config_filename = model_path.join("config.json");
            let tokenizer_filename = model_path.join("tokenizer.json");
            let weights_filename = if self.use_pth {
                model_path.join("pytorch_model.bin")
            } else {
                model_path.join("model.safetensors")
            };
            (config_filename, tokenizer_filename, weights_filename)


        } else {
            let repo = Repo::with_revision(model_id, RepoType::Model, revision);
            
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = if self.use_pth {
                api.get("pytorch_model.bin")?
            } else {
                api.get("model.safetensors")?
            };
            (config, tokenizer, weights)
        };


        let config = std::fs::read_to_string(config_filename)?;
        let mut config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb = if self.use_pth {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };
        if self.approximate_gelu {
            config.hidden_act = HiddenAct::GeluApproximate;
        }
        let model = BertModelWithClassifier::load(vb, &config)?;
        Ok((model, tokenizer, config))
    }
}



fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    let start = std::time::Instant::now();

    let (model, mut tokenizer, config) = args.build_model_and_tokenizer_config()?;
    let device = &model.device;

    if let Some(prompt) = args.prompt {
        let tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)?;
        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;
        println!("Loaded and encoded {:?}", start.elapsed());
        let id2label = config.get_id2labels()
                                                        .as_ref()
                                                        .unwrap();
        for idx in 0..args.n {
            let start = std::time::Instant::now();
            let ys = model.forward(&token_ids, &token_type_ids)?;
            // println!("{:?}", tokens);
            let ys_vec = ys.argmax(2)?
                        .squeeze(0).unwrap()
                        .to_vec1::<u32>()?;
            // println!("{:?}", ys_vec);
            

            for (y_pred, sent) in ys_vec.iter().zip(tokens.iter()) {
                    let label = id2label.get(& y_pred.to_string()).unwrap();
                    let token = tokenizer.id_to_token(*sent).unwrap();
                    println!("{:?}<==>{:?}", label, token);
                // }
            }
            
            // if idx == 0 {
            //     println!("{ys}");
            // }
            println!("Took {:?}", start.elapsed());
             }
            }
        Ok(())
    }
   

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
