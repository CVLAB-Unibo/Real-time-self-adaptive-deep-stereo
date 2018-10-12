#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>

REGISTER_OP("ShiftCorr")
    .Input("in0: float")
    .Input("in1: float")
    .Attr("max_disp: int")
    .Output("output: float");

REGISTER_OP("ShiftCorrGrad")
    .Input("in0: float")
    .Input("in1: float")
    .Input("grad: float")
    .Attr("max_disp: int")
    .Output("output0: float")
    .Output("output1: float");


using namespace tensorflow;

void ShiftCorrKernelLauncher(const float *vals0, const float *vals1, const int max_disp,
        const int batch_size, const int n_channels, const int in_h, const int in_w, float *out);

class ShiftCorrOp : public OpKernel {
public:
    int max_disp_;
    explicit ShiftCorrOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context,
                    context->GetAttr("max_disp", &max_disp_));
    }

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& in0 = context->input(0);
        const Tensor& in1 = context->input(1);
        auto values0 = in0.flat<float>();
        auto values1 = in1.flat<float>();

        // Create an output tensor
        TensorShape in_shape = in0.shape();  // input in NHWC
        TensorShape out_shape = in_shape;
        out_shape.set_dim(1, max_disp_ * 2 + 1);
        out_shape.set_dim(2, in_shape.dim_size(1));
        out_shape.set_dim(3, in_shape.dim_size(2) - 2*max_disp_);  // output in NCHW
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                    &output_tensor));
        auto output = output_tensor->flat<float>();

        ShiftCorrKernelLauncher(values0.data(), values1.data(), max_disp_,
                    in_shape.dim_size(0), in_shape.dim_size(1), in_shape.dim_size(2),
                    in_shape.dim_size(3), output.data());
    }
};
REGISTER_KERNEL_BUILDER(Name("ShiftCorr").Device(DEVICE_GPU), ShiftCorrOp);

void ShiftCorrGradKernelLauncher(const float *input0, const float *input1, const float *grad,
        const int max_disp_, const int batch_size, const int height, const int width,
        const int channels, float *out0, float *out1);

class ShiftCorrGradOp : public OpKernel {
public:
    int max_disp_;
    explicit ShiftCorrGradOp(OpKernelConstruction* context) : OpKernel(context) {
	OP_REQUIRES_OK(context,
                    context->GetAttr("max_disp", &max_disp_));
    }

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& in0 = context->input(0);
        const Tensor& in1 = context->input(1);
        const Tensor& grad = context->input(2);
        auto values0 = in0.flat<float>();
        auto values1 = in0.flat<float>();
        auto grad_values = grad.flat<float>();

        // Create an output tensor
        Tensor* out0 = nullptr;
        Tensor* out1 = nullptr;
        TensorShape in_shape = in0.shape();
        OP_REQUIRES_OK(context, context->allocate_output(0, in_shape, &out0));
        OP_REQUIRES_OK(context, context->allocate_output(1, in_shape, &out1));
        auto out_values0 = out0->flat<float>();
        auto out_values1 = out1->flat<float>();

        ShiftCorrGradKernelLauncher(values0.data(), values1.data(), grad_values.data(),
                    max_disp_, in_shape.dim_size(0), in_shape.dim_size(1), in_shape.dim_size(2),
                    in_shape.dim_size(3), out_values0.data(), out_values1.data());
    }
};
REGISTER_KERNEL_BUILDER(Name("ShiftCorrGrad").Device(DEVICE_GPU), ShiftCorrGradOp);
