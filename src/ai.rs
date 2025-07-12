use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::Backend;
use burn::tensor::activation::{sigmoid, softmax};
use burn::tensor::{Bool, Distribution, Tensor};
use rand::Rng;

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

    fn interleave_bw_linear(a: &Linear<B>, b: &Linear<B>, device: &B::Device) -> Linear<B> {
        Linear {
            weight: Param::from_tensor(Self::interleave(
                a.weight.clone().into_value(),
                b.weight.clone().into_value(),
                device,
            )),
            bias: b.bias.clone(),
        }
    }

    pub fn interleave(a: Tensor<B, 2>, b: Tensor<B, 2>, device: &B::Device) -> Tensor<B, 2> {
        let a_size = a.shape().dims[0];
        let b_size = b.shape().dims[0];
        assert_eq!(a_size, b_size);

        let mut rng = rand::rng();
        let a_mask: u128 = rng.random();
        let b_mask = !a_mask;

        let a_mask_arr = (0..a_size)
            .map(|i| (a_mask >> i) & 1 == 1)
            .collect::<Vec<_>>();

        let b_mask_arr = (0..a_size)
            .map(|i| (b_mask >> i) & 1 == 1)
            .collect::<Vec<_>>();

        let a_vals = a.clone().mask_where(
            Tensor::<B, 1, Bool>::from_data(a_mask_arr.as_slice(), device),
            a.zeros_like(),
        );
        let b_vals = b.clone().mask_where(
            Tensor::<B, 1, Bool>::from_data(b_mask_arr.as_slice(), device),
            b.zeros_like(),
        );

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

    pub fn apply(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        let x = sigmoid(self.input.forward(input));
        let x = sigmoid(self.hidden_1.forward(x));
        let x = sigmoid(self.hidden_2.forward(x));
        let x = softmax(self.hidden_3.forward(x), 0);
        let x = sigmoid(self.output.forward(x));
        x
    }
}
