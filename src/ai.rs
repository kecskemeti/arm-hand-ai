use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::activation::{relu, tanh};
use burn::tensor::Distribution::Uniform;
use burn::tensor::{Distribution, Tensor};
use std::ops::Deref;

#[derive(Module, Debug)]
pub struct AI<B: Backend> {
    input: Linear<B>,
    output: Linear<B>,
    hidden_1: Linear<B>,
    hidden_2: Linear<B>,
    hidden_3: Linear<B>,
}

impl<B: Backend> AI<B> {
    pub fn new(device: &B::Device) -> Self {
        let input_config = LinearConfig::new(64, 256)
            .with_bias(true)
            .with_initializer(Initializer::Normal { mean: 0., std: 1. });

        let output_config = LinearConfig::new(32, 7)
            .with_bias(true)
            .with_initializer(Initializer::Normal { mean: 0., std: 1. });

        let hidden_1_config = LinearConfig::new(256, 128)
            .with_bias(true)
            .with_initializer(Initializer::Normal { mean: 0., std: 1. });

        let hidden_2_config = LinearConfig::new(128, 64)
            .with_bias(true)
            .with_initializer(Initializer::Normal { mean: 0., std: 1. });

        let hidden_3_config = LinearConfig::new(64, 32)
            .with_bias(true)
            .with_initializer(Initializer::Normal { mean: 0., std: 1. });

        Self {
            input: input_config.init(device),
            output: output_config.init(device),
            hidden_1: hidden_1_config.init(device),
            hidden_2: hidden_2_config.init(device),
            hidden_3: hidden_3_config.init(device),
        }
    }

    fn jiggle_tensor<const N: usize>(t: &Tensor<B, N>, d: &Distribution) -> Tensor<B, N> {
        let jiggle_with = t.random_like(*d);
        t.clone().add(jiggle_with)
    }
    fn jiggle_linear(ln: &Linear<B>, d: &Distribution) -> Linear<B> {
        Linear {
            weight: Param::from_tensor(Self::jiggle_tensor(&ln.weight, d)),
            bias: ln
                .bias
                .as_ref()
                .map(|p| Param::from_tensor(Self::jiggle_tensor(p, d))),
        }
    }
    pub fn jiggle(&self, d: &Distribution) -> Self {
        Self {
            input: Self::jiggle_linear(&self.input, &d),
            output: Self::jiggle_linear(&self.output, &d),
            hidden_1: Self::jiggle_linear(&self.hidden_1, &d),
            hidden_2: Self::jiggle_linear(&self.hidden_2, &d),
            hidden_3: Self::jiggle_linear(&self.hidden_3, &d),
        }
    }

    fn combine_bw_linear(a: &Linear<B>, b: &Linear<B>) -> Linear<B> {
        Linear {
            weight: a.weight.clone(),
            bias: b.bias.clone(),
        }
    }

    fn interleave_bw_linear(a: &Linear<B>, b: &Linear<B>) -> Linear<B> {
        Linear {
            weight: Param::from_tensor(Self::interleave(
                a.weight.deref().clone(),
                b.weight.deref().clone(),
            )),
            bias: b
                .bias
                .iter()
                .flat_map(|bp| {
                    a.bias
                        .as_ref()
                        .map(|ap| Self::interleave(ap.deref().clone(), bp.deref().clone()))
                })
                .map(Param::from_tensor)
                .next(),
        }
    }

    pub fn interleave<const N: usize>(a: Tensor<B, N>, b: Tensor<B, N>) -> Tensor<B, N> {
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

    pub fn average<const N: usize>(a: Tensor<B, N>, b: Tensor<B, N>) -> Tensor<B, N> {
        (a + b) / 2.
    }

    fn average_bw_linear(a: &Linear<B>, b: &Linear<B>) -> Linear<B> {
        Linear {
            weight: Param::from_tensor(Self::average(
                a.weight.deref().clone(),
                b.weight.deref().clone(),
            )),
            bias: b
                .bias
                .iter()
                .flat_map(|bp| {
                    a.bias
                        .as_ref()
                        .map(|ap| Self::average(ap.deref().clone(), bp.deref().clone()))
                })
                .map(Param::from_tensor)
                .next(),
        }
    }

    pub fn offspring(&self, other_parent: &Self, d: &Distribution) -> Self {
        Self {
            input: Self::combine_bw_linear(&self.input, &other_parent.input),
            output: Self::combine_bw_linear(&self.output, &other_parent.output),
            hidden_1: Self::combine_bw_linear(&self.hidden_1, &other_parent.hidden_1),
            hidden_2: Self::combine_bw_linear(&self.hidden_2, &other_parent.hidden_2),
            hidden_3: Self::combine_bw_linear(&self.hidden_3, &other_parent.hidden_3),
        }
        .jiggle(d)
    }

    pub fn offspring_iw(&self, other_parent: &Self, d: &Distribution) -> Self {
        Self {
            input: Self::interleave_bw_linear(&self.input, &other_parent.input),
            output: Self::interleave_bw_linear(&self.output, &other_parent.output),
            hidden_1: Self::interleave_bw_linear(&self.hidden_1, &other_parent.hidden_1),
            hidden_2: Self::interleave_bw_linear(&self.hidden_2, &other_parent.hidden_2),
            hidden_3: Self::interleave_bw_linear(&self.hidden_3, &other_parent.hidden_3),
        }
        .jiggle(d)
    }

    pub fn offspring_aw(&self, other_parent: &Self, d: &Distribution) -> Self {
        Self {
            input: Self::average_bw_linear(&self.input, &other_parent.input),
            output: Self::average_bw_linear(&self.output, &other_parent.output),
            hidden_1: Self::average_bw_linear(&self.hidden_1, &other_parent.hidden_1),
            hidden_2: Self::average_bw_linear(&self.hidden_2, &other_parent.hidden_2),
            hidden_3: Self::average_bw_linear(&self.hidden_3, &other_parent.hidden_3),
        }
        .jiggle(d)
    }

    pub fn offspring_layers(&self, other_parent: &Self, d: &Distribution) -> Self {
        Self {
            input: self.input.clone(),
            output: other_parent.output.clone(),
            hidden_1: self.hidden_1.clone(),
            hidden_2: other_parent.hidden_2.clone(),
            hidden_3: self.hidden_3.clone(),
        }
        .jiggle(d)
    }

    pub fn apply(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        let x = relu(self.input.forward(input));
        let x = relu(self.hidden_1.forward(x));
        let x = relu(self.hidden_2.forward(x));
        let x = relu(self.hidden_3.forward(x));
        let x = tanh(self.output.forward(x));
        x
    }

    pub fn max_amp_for_tensor<const N: usize>(input: &Tensor<B, N>) -> f32 {
        let data = input.clone().to_data();
        let slice: &[f32] = data.as_slice().unwrap();

        slice
            .iter()
            .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
            .copied()
            .unwrap()
            .abs()
    }

    pub fn max_amp_for_linear(input: &Linear<B>) -> f32 {
        let weight_max = Self::max_amp_for_tensor(&input.weight);
        let bias_max = Self::max_amp_for_tensor(input.bias.as_ref().unwrap());
        weight_max.max(bias_max)
    }
    pub fn max_amp(&self) -> f32 {
        let all_maximums = [
            Self::max_amp_for_linear(&self.input),
            Self::max_amp_for_linear(&self.hidden_1),
            Self::max_amp_for_linear(&self.hidden_2),
            Self::max_amp_for_linear(&self.hidden_3),
            Self::max_amp_for_linear(&self.output),
        ];
        all_maximums
            .iter()
            .max_by(|a, b| a.partial_cmp(&b).unwrap())
            .copied()
            .unwrap()
    }
}
