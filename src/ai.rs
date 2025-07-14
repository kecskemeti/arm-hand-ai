use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::activation::{relu, tanh};
use burn::tensor::Distribution::Uniform;
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
    pub fn jiggle(&self) -> Self {
        // distribution should be a param!
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

    fn interleave_bw_linear(a: &Linear<B>, b: &Linear<B>) -> Linear<B> {
        Linear {
            weight: Param::from_tensor(Self::interleave(
                a.weight.clone().into_value(),
                b.weight.clone().into_value(),
            )),
            bias: b
                .bias
                .as_ref()
                .map(|p| Param::from_tensor(p.clone().into_value())),
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

    pub fn offspring_iw(&self, other_parent: &Self) -> Self {
        Self {
            input: Self::interleave_bw_linear(&self.input, &other_parent.input),
            output: Self::interleave_bw_linear(&self.output, &other_parent.output),
            hidden_1: Self::interleave_bw_linear(&self.hidden_1, &other_parent.hidden_1),
            hidden_2: Self::interleave_bw_linear(&self.hidden_2, &other_parent.hidden_2),
            hidden_3: Self::interleave_bw_linear(&self.hidden_3, &other_parent.hidden_3),
        }
        .jiggle()
    }

    // create an averaging offspring

    // create an offspring when we change the layers
    // linear 1 is coming from parent 1
    // linear 2...
    // combine

    pub fn apply(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        let x = relu(self.input.forward(input));
        let x = relu(self.hidden_1.forward(x));
        let x = relu(self.hidden_2.forward(x));
        let x = relu(self.hidden_3.forward(x));
        let x = tanh(self.output.forward(x));
        x
    }
}
