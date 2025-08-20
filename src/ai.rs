use crate::base_ai::{
    average_bw_linear, combine_bw_linear, interleave_bw_linear, jiggle_linear, max_amp_for_linear,
    AI,
};
use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::activation::{relu, tanh};
use burn::tensor::{Distribution, Tensor};

#[derive(Module, Debug)]
pub struct BigAI<B: Backend> {
    input: Linear<B>,
    output: Linear<B>,
    hidden_1: Linear<B>,
    hidden_2: Linear<B>,
    hidden_3: Linear<B>,
}

impl<B: Backend> AI<B> for BigAI<B> {
    fn jiggle(&self, d: &Distribution) -> Self {
        Self {
            input: jiggle_linear(&self.input, &d),
            output: jiggle_linear(&self.output, &d),
            hidden_1: jiggle_linear(&self.hidden_1, &d),
            hidden_2: jiggle_linear(&self.hidden_2, &d),
            hidden_3: jiggle_linear(&self.hidden_3, &d),
        }
    }

    fn offspring(&self, other_parent: &Self, d: &Distribution) -> Self {
        Self {
            input: combine_bw_linear(&self.input, &other_parent.input),
            output: combine_bw_linear(&self.output, &other_parent.output),
            hidden_1: combine_bw_linear(&self.hidden_1, &other_parent.hidden_1),
            hidden_2: combine_bw_linear(&self.hidden_2, &other_parent.hidden_2),
            hidden_3: combine_bw_linear(&self.hidden_3, &other_parent.hidden_3),
        }
        .jiggle(d)
    }

    fn offspring_iw(&self, other_parent: &Self, d: &Distribution) -> Self {
        Self {
            input: interleave_bw_linear(&self.input, &other_parent.input),
            output: interleave_bw_linear(&self.output, &other_parent.output),
            hidden_1: interleave_bw_linear(&self.hidden_1, &other_parent.hidden_1),
            hidden_2: interleave_bw_linear(&self.hidden_2, &other_parent.hidden_2),
            hidden_3: interleave_bw_linear(&self.hidden_3, &other_parent.hidden_3),
        }
        .jiggle(d)
    }

    fn offspring_aw(&self, other_parent: &Self, d: &Distribution) -> Self {
        Self {
            input: average_bw_linear(&self.input, &other_parent.input),
            output: average_bw_linear(&self.output, &other_parent.output),
            hidden_1: average_bw_linear(&self.hidden_1, &other_parent.hidden_1),
            hidden_2: average_bw_linear(&self.hidden_2, &other_parent.hidden_2),
            hidden_3: average_bw_linear(&self.hidden_3, &other_parent.hidden_3),
        }
        .jiggle(d)
    }

    fn offspring_layers(&self, other_parent: &Self, d: &Distribution) -> Self {
        Self {
            input: self.input.clone(),
            output: other_parent.output.clone(),
            hidden_1: self.hidden_1.clone(),
            hidden_2: other_parent.hidden_2.clone(),
            hidden_3: self.hidden_3.clone(),
        }
        .jiggle(d)
    }

    fn apply(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        let x = relu(self.input.forward(input));
        let x = relu(self.hidden_1.forward(x));
        let x = relu(self.hidden_2.forward(x));
        let x = relu(self.hidden_3.forward(x));
        let x = tanh(self.output.forward(x));
        x
    }

    fn max_amp(&self) -> f32 {
        let all_maximums = [
            max_amp_for_linear(&self.input),
            max_amp_for_linear(&self.hidden_1),
            max_amp_for_linear(&self.hidden_2),
            max_amp_for_linear(&self.hidden_3),
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

    fn save_file(&self, filename: &str, recorder: &NamedMpkFileRecorder<FullPrecisionSettings>) {
        self.clone()
            .save_file(filename, recorder)
            .expect("save failed");
    }

    fn load_a_file(
        self,
        filename: &str,
        recorder: &NamedMpkFileRecorder<FullPrecisionSettings>,
    ) -> Self {
        let device = self.input.devices()[0].clone();
        self.load_file(filename, recorder, &device)
            .expect("load failed")
    }

    fn network_name(&self) -> &'static str {
        "BigAI"
    }
}

impl<B: Backend> BigAI<B> {
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
}
