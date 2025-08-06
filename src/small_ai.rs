use crate::base_ai::{
    average_bw_linear, combine_bw_linear, interleave_bw_linear, jiggle_linear, max_amp_for_linear,
    AI,
};
use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::activation::{relu, tanh};
use burn::tensor::{Distribution, Tensor};

#[derive(Module, Debug)]
pub struct SmallAI<B: Backend> {
    input: Linear<B>,
    output: Linear<B>,
    hidden: Linear<B>,
}
impl<B: Backend> AI<B> for SmallAI<B> {
    fn jiggle(&self, d: &Distribution) -> Self {
        Self {
            input: jiggle_linear(&self.input, &d),
            output: jiggle_linear(&self.output, &d),
            hidden: jiggle_linear(&self.hidden, &d),
        }
    }
    fn offspring(&self, other_parent: &Self, d: &Distribution) -> Self {
        Self {
            input: combine_bw_linear(&self.input, &other_parent.input),
            output: combine_bw_linear(&self.output, &other_parent.output),
            hidden: combine_bw_linear(&self.hidden, &other_parent.hidden),
        }
        .jiggle(d)
    }

    fn offspring_iw(&self, other_parent: &Self, d: &Distribution) -> Self {
        Self {
            input: interleave_bw_linear(&self.input, &other_parent.input),
            output: interleave_bw_linear(&self.output, &other_parent.output),
            hidden: interleave_bw_linear(&self.hidden, &other_parent.hidden),
        }
        .jiggle(d)
    }

    fn offspring_aw(&self, other_parent: &Self, d: &Distribution) -> Self {
        Self {
            input: average_bw_linear(&self.input, &other_parent.input),
            output: average_bw_linear(&self.output, &other_parent.output),
            hidden: average_bw_linear(&self.hidden, &other_parent.hidden),
        }
        .jiggle(d)
    }

    fn offspring_layers(&self, other_parent: &Self, d: &Distribution) -> Self {
        Self {
            input: self.input.clone(),
            output: other_parent.output.clone(),
            hidden: self.hidden.clone(),
        }
        .jiggle(d)
    }

    fn apply(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        let x = relu(self.input.forward(input));
        let x = relu(self.hidden.forward(x));
        let x = tanh(self.output.forward(x));
        x
    }

    fn max_amp(&self) -> f32 {
        let all_maximums = [
            max_amp_for_linear(&self.input),
            max_amp_for_linear(&self.hidden),
            max_amp_for_linear(&self.output),
        ];
        all_maximums
            .iter()
            .max_by(|a, b| {
                a.partial_cmp(&b)
                    .expect("max amplitude comparison failed across all layers")
            })
            .copied()
            .expect("no max amplitude found across all layers")
    }
}

impl<B: Backend> SmallAI<B> {
    pub fn new(device: &B::Device) -> Self {
        let input_config = LinearConfig::new(64, 128)
            .with_bias(true)
            .with_initializer(Initializer::Normal { mean: 0., std: 1. });

        let output_config = LinearConfig::new(14, 7)
            .with_bias(true)
            .with_initializer(Initializer::Normal { mean: 0., std: 1. });

        let hidden_config = LinearConfig::new(128, 14)
            .with_bias(true)
            .with_initializer(Initializer::Normal { mean: 0., std: 1. });

        Self {
            input: input_config.init(device),
            output: output_config.init(device),
            hidden: hidden_config.init(device),
        }
    }
}
