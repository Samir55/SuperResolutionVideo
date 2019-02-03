/*
MIT License

Copyright (c) 2018 Adrian Köring

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

 */

#pragma once

#include <string>
#include "tensorflow/include/tensorflow/c/c_api.h"


namespace tensorflow_cc_inference
{

class Inference {

private:
	TF_Graph*   graph;
	TF_Session* session;

	TF_Operation* input_op;
	TF_Output 		input;

	TF_Operation* output_op;
	TF_Output 		output;

	/**
	 * Load a protobuf buffer from disk,
	 * recreate the tensorflow graph and
	 * provide it for inference.
	 */
	TF_Buffer* ReadBinaryProto(const std::string& fname) const;

	/**
	 * Tensorflow does not throw errors but manages runtime information
	 *   in a _Status_ object containing error codes and a failure message.
	 *
	 * AssertOk throws a runtime_error if Tensorflow communicates an
	 *   exceptional status.
	 *
	 */
	void AssertOk(const TF_Status* status) const ;

public:
	/**
	 * binary_graphdef_protobuf_filename: only binary protobuffers
	 *   seem to be supported via the tensorflow C api.
	 * input_node_name: the name of the node that should be feed with the
	 *   input tensor
	 * output_node_name: the node from which the output tensor should be
	 *   retrieved
	 */
	Inference(const std::string& binary_graphdef_protobuf_filename,
						const std::string& input_node_name,
						const std::string& output_node_name);

	/**
	 * Clean up all pointer-members using the dedicated tensorflor api functions
	 */
	~Inference();

	/**
	 * Run the graph on some input data.
	 *
	 * Provide the input and output tensor.
	 */
	 TF_Tensor* operator()(TF_Tensor* input_tensor) const;

};

} // namespace tensorflow_cc_inference
