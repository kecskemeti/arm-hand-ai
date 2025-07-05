use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::{Distribution, Tensor};

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
        let input_config = LinearConfig::new(64, 128)
            .with_bias(true)
            .with_initializer(Initializer::Normal { mean: 0., std: 1. });

        let output_config = LinearConfig::new(32, 7)
            .with_bias(true)
            .with_initializer(Initializer::Normal { mean: 0., std: 1. });

        let hidden_1_config = LinearConfig::new(128, 128)
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
    fn jiggle(&self) -> Self {
        let d = Distribution::Normal(0., 0.0001);
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

    pub fn offspring(&self, other_parent: &Self) -> Self {
        Self {
            input: Self::combine_bw_linear(&self.input, &other_parent.input),
            output: Self::combine_bw_linear(&self.output, &other_parent.output),
            hidden_1: Self::combine_bw_linear(&self.hidden_1, &other_parent.hidden_1),
            hidden_2: Self::combine_bw_linear(&self.hidden_2, &other_parent.hidden_2),
            hidden_3: Self::combine_bw_linear(&self.hidden_3, &other_parent.hidden_3),
        }
        .jiggle()
    }

    pub fn apply(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        let x = sigmoid(self.input.forward(input));
        let x = sigmoid(self.hidden_1.forward(x));
        let x = sigmoid(self.hidden_2.forward(x));
        let x = softmax(self.hidden_3.forward(x), 0);
        let x = sigmoid(self.output.forward(x));
        x
    }
}
