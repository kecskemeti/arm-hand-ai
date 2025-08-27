use burn::module::{Module, Param};
use burn::nn::Linear;
use burn::prelude::Backend;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::Distribution::Uniform;
use burn::tensor::{Distribution, Tensor};
use regex::Regex;
use std::fmt::Debug;
use std::fs;
use std::ops::Deref;
use std::sync::LazyLock;

pub trait AI<B: Backend>: Module<B> + Debug {
    fn jiggle(&self, d: &Distribution) -> Self;
    fn offspring(&self, other_parent: &Self, d: &Distribution) -> Self;
    fn offspring_iw(&self, other_parent: &Self, d: &Distribution) -> Self;
    fn offspring_aw(&self, other_parent: &Self, d: &Distribution) -> Self;
    fn offspring_layers(&self, other_parent: &Self, d: &Distribution) -> Self;
    fn apply(&self, input: Tensor<B, 1>) -> Tensor<B, 1>;
    fn max_amp(&self) -> f32;

    fn save_file(&self, filename: &str, recorder: &NamedMpkFileRecorder<FullPrecisionSettings>);
    fn load_a_file(
        self,
        filename: &str,
        recorder: &NamedMpkFileRecorder<FullPrecisionSettings>,
    ) -> Self;

    fn network_name(&self) -> &'static str;
}

fn jiggle_tensor<const N: usize, B: Backend>(t: &Tensor<B, N>, d: &Distribution) -> Tensor<B, N> {
    let jiggle_with = t.random_like(*d);
    t.clone().add(jiggle_with)
}
pub fn jiggle_linear<B: Backend>(ln: &Linear<B>, d: &Distribution) -> Linear<B> {
    Linear {
        weight: Param::from_tensor(jiggle_tensor(&ln.weight, d)),
        bias: ln
            .bias
            .as_ref()
            .map(|p| Param::from_tensor(jiggle_tensor(p, d))),
    }
}

pub fn combine_bw_linear<B: Backend>(a: &Linear<B>, b: &Linear<B>) -> Linear<B> {
    Linear {
        weight: a.weight.clone(),
        bias: b.bias.clone(),
    }
}

pub fn interleave_bw_linear<B: Backend>(a: &Linear<B>, b: &Linear<B>) -> Linear<B> {
    Linear {
        weight: Param::from_tensor(interleave(
            a.weight.deref().clone(),
            b.weight.deref().clone(),
        )),
        bias: b
            .bias
            .iter()
            .flat_map(|bp| {
                a.bias
                    .as_ref()
                    .map(|ap| interleave(ap.deref().clone(), bp.deref().clone()))
            })
            .map(Param::from_tensor)
            .next(),
    }
}

pub fn interleave<const N: usize, B: Backend>(a: Tensor<B, N>, b: Tensor<B, N>) -> Tensor<B, N> {
    let a_size = a.shape();
    let b_size = b.shape();
    assert_eq!(a_size, b_size);

    let baseline = a.random_like(Uniform(0., 1.));
    let a_mask = baseline.clone().lower_elem(0.5);
    let b_mask = baseline.greater_equal_elem(0.5);

    let a_vals = a.clone().mask_where(a_mask, a.zeros_like());
    let b_vals = b.clone().mask_where(b_mask, b.zeros_like());

    a_vals + b_vals
}

pub fn average<const N: usize, B: Backend>(a: Tensor<B, N>, b: Tensor<B, N>) -> Tensor<B, N> {
    (a + b) / 2.
}

pub fn average_bw_linear<B: Backend>(a: &Linear<B>, b: &Linear<B>) -> Linear<B> {
    Linear {
        weight: Param::from_tensor(average(a.weight.deref().clone(), b.weight.deref().clone())),
        bias: b
            .bias
            .iter()
            .flat_map(|bp| {
                a.bias
                    .as_ref()
                    .map(|ap| average(ap.deref().clone(), bp.deref().clone()))
            })
            .map(Param::from_tensor)
            .next(),
    }
}

pub fn max_amp_for_tensor<const N: usize, B: Backend>(input: &Tensor<B, N>) -> f32 {
    let data = input.clone().to_data();
    let slice: &[f32] = data
        .as_slice()
        .expect("tensor is not representable as a slice");

    slice
        .iter()
        .max_by(|a, b| {
            a.abs()
                .partial_cmp(&b.abs())
                .expect("amplitude comparison failed")
        })
        .copied()
        .expect("no max amplitude found")
        .abs()
}

pub fn max_amp_for_linear<B: Backend>(input: &Linear<B>) -> f32 {
    let weight_max = max_amp_for_tensor(&input.weight);
    let bias_max = max_amp_for_tensor(input.bias.as_ref().expect("bias not present"));
    weight_max.max(bias_max)
}

pub trait ListableAI<B: Backend>: AI<B> {
    fn list(&self) -> Vec<String>;
}

impl<B: Backend, A: AI<B>> ListableAI<B> for A {
    fn list(&self) -> Vec<String> {
        let network_name = self.network_name();
        let mut saved_files = Vec::new();

        // Get the current directory (where model files are saved)
        if let Ok(entries) = fs::read_dir(".") {
            for entry in entries.flatten() {
                if let Some(filename) = entry.file_name().to_str() {
                    if let Some(seq) = extract_seq(filename, network_name) {
                        saved_files.push((seq, filename.to_string()));
                    }
                }
            }
        }

        saved_files.sort_by_key(|a| a.0);

        saved_files
            .into_iter()
            .rev()
            .take(30)
            .map(|(_, filename)| filename)
            .collect()
    }
}

static REG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Za-z]+_(?<net_name>[A-Z a-z]+)_(?<seq>[0-9]+)\.mpk").unwrap());

pub fn extract_seq(filename: &str, network_name: &str) -> Option<usize> {
    REG.captures(filename)
        .filter(|c| &c["net_name"] == network_name)
        .map(|c| c.name("seq").map(|m| m.as_str()))
        .flatten()
        .map(|seq_str| seq_str.parse().ok())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::small_ai::SmallAI;
    use burn::backend::candle::CandleDevice;
    use burn::backend::Candle;

    #[test]
    fn test_extract_seq() {
        assert_eq!(extract_seq("best_te st_1234.mpk", "te st"), Some(1234));
    }

    #[test]
    fn test_load_fnames() {
        type BE = Candle<f32, i64>;

        let device = CandleDevice::Cpu;

        let small_ai = SmallAI::<BE>::new(&device);

        let fnames = small_ai.list();
        println!("{:?}", fnames);
    }
}
