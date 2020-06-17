import cupy
import torch
import re

kernel_Sepconv_updateOutput = '''
    extern "C" __global__ void kernel_Sepconv_updateOutput(
        const int n,
        const float* input,
        const float* vertical,
        const float* horizontal,
        float* output
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float dblOutput = 0.0;

        const int intSample = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
        const int intDepth  = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
        const int intY      = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
        const int intX      = ( intIndex                                                    ) % SIZE_3(output);

        for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1) {
            for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1) {
                dblOutput += VALUE_4(input, intSample, intDepth, intY + intFilterY, intX + intFilterX) 
                             * VALUE_4(vertical, intSample, intFilterY, intY, intX) 
                             * VALUE_4(horizontal, intSample, intFilterX, intY, intX);
            }
        }

        output[intIndex] = dblOutput;
    } }
'''

kernel_Sepconv_updateGradInput = '''
	extern "C" __global__ void kernel_Sepconv_updateGradInput(
		const int n,
		const float* vertical,
		const float* horizontal,
		const float* gradOutput,
		float* gradInput
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float dInput = 0.0;

		const int intSample = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput) / SIZE_1(gradInput)	) % SIZE_0(gradInput);
		const int intDepth  = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput)                  		) % SIZE_1(gradInput);
		const int intY      = ( intIndex / SIZE_3(gradInput)                                   			) % SIZE_2(gradInput);
		const int intX      = ( intIndex                                                    			) % SIZE_3(gradInput);
		const int maxOutY 	= SIZE_2(gradOutput);
		const int maxOutX 	= SIZE_3(gradOutput);

		for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1) {
			if (intY - intFilterY < 0){ break; }
			if (intY - intFilterY > maxOutY){ continue; }
			for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1) {
				if (intX - intFilterX < 0){ break; }
				if (intX - intFilterX > maxOutX){ continue; }
				dInput += VALUE_4(gradOutput, intSample, intDepth, intY - intFilterY, intX - intFilterX)
						  * VALUE_4(vertical, intSample, intFilterY, intY - intFilterY, intX - intFilterX) 
						  * VALUE_4(horizontal, intSample, intFilterX, intY - intFilterY, intX - intFilterX);
			}
		}

		gradInput[intIndex] = dInput;
	} }
'''

# kernel_Sepconv_updateGradInput = '''
#     extern "C" __global__ void kernel_Sepconv_updateGradInput(
#         const int ni, const int nv, const int nh,
#         const float* input,
#         const float* vertical,
#         const float* horizontal,
#         const float* gradOutput,
#         float* gradInput,
#         float* gradVertical,
#         float* gradHorizontal
#     ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < ni; intIndex += blockDim.x * gridDim.x) {
#         float dInput = 0.0;

#         const int intSample = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput) / SIZE_1(gradInput)	) % SIZE_0(gradInput);
#         const int intDepth  = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput)                  		) % SIZE_1(gradInput);
#         const int intY      = ( intIndex / SIZE_3(gradInput)                                   			) % SIZE_2(gradInput);
#         const int intX      = ( intIndex                                                    			) % SIZE_3(gradInput);
#         const int maxOutY 	= SIZE_2(gradOutput);
#         const int maxOutX 	= SIZE_3(gradOutput);

#         for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1) {
#             if (intY - intFilterY < 0){ break; }
#             if (intY - intFilterY > maxOutY){ continue; }
#             for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1) {
#                 if (intX - intFilterX < 0){ break; }
#                 if (intX - intFilterX > maxOutX){ continue; }
#                 dInput += VALUE_4(gradOutput, intSample, intDepth, intY - intFilterY, intX - intFilterX)
#                           * VALUE_4(vertical, intSample, intFilterY, intY - intFilterY, intX - intFilterX) 
#                           * VALUE_4(horizontal, intSample, intFilterX, intY - intFilterY, intX - intFilterX);
#             }
#         }

#         gradInput[intIndex] = dInput;
#     }

#         for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < nv; intIndex += blockDim.x * gridDim.x) {
#         float dVertical = 0.0;

#         const int intSample 	= ( intIndex / SIZE_3(gradVertical) / SIZE_2(gradVertical) / SIZE_1(gradVertical) 	) % SIZE_0(gradVertical);
#         const int intFilterY  	= ( intIndex / SIZE_3(gradVertical) / SIZE_2(gradVertical)                  		) % SIZE_1(gradVertical);
#         const int intY      	= ( intIndex / SIZE_3(gradVertical)                                   				) % SIZE_2(gradVertical);
#         const int intX      	= ( intIndex                                                    					) % SIZE_3(gradVertical);

#         for (int intDepth = 0; intDepth < SIZE_1(input); intDepth += 1) {
#             for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1) {
#                 dVertical += VALUE_4(gradOutput, intSample, intDepth, intY, intX)
#                              * VALUE_4(input, intSample, intDepth, intY + intFilterY, intX + intFilterX) 
#                              * VALUE_4(horizontal, intSample, intFilterX, intY, intX);
#             }
#         }

#         gradVertical[intIndex] = dVertical;
#     }
#         for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < nh; intIndex += blockDim.x * gridDim.x) {
#         float dHorizontal = 0.0;

#         const int intSample 	= ( intIndex / SIZE_3(gradHorizontal) / SIZE_2(gradHorizontal) / SIZE_1(gradHorizontal) ) % SIZE_0(gradHorizontal);
#         const int intFilterX  	= ( intIndex / SIZE_3(gradHorizontal) / SIZE_2(gradHorizontal)                  		) % SIZE_1(gradHorizontal);
#         const int intY      	= ( intIndex / SIZE_3(gradHorizontal)                                   				) % SIZE_2(gradHorizontal);
#         const int intX      	= ( intIndex                                                    						) % SIZE_3(gradHorizontal);

#         for (int intDepth = 0; intDepth < SIZE_1(input); intDepth += 1) {
#             for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1) {
#                 dHorizontal += VALUE_4(gradOutput, intSample, intDepth, intY, intX)
#                                * VALUE_4(input, intSample, intDepth, intY + intFilterY, intX + intFilterX) 
#                                * VALUE_4(vertical, intSample, intFilterY, intY, intX);
#             }
#         }

#         gradHorizontal[intIndex] = dHorizontal;
#     } }
# '''

kernel_Sepconv_updateGradVertical = '''
    extern "C" __global__ void kernel_Sepconv_updateGradVertical(
        const int n,
        const float* input,
        const float* horizontal,
        const float* gradOutput,
        float* gradVertical
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float dVertical = 0.0;

        const int intSample 	= ( intIndex / SIZE_3(gradVertical) / SIZE_2(gradVertical) / SIZE_1(gradVertical) 	) % SIZE_0(gradVertical);
        const int intFilterY  	= ( intIndex / SIZE_3(gradVertical) / SIZE_2(gradVertical)                  		) % SIZE_1(gradVertical);
        const int intY      	= ( intIndex / SIZE_3(gradVertical)                                   				) % SIZE_2(gradVertical);
        const int intX      	= ( intIndex                                                    					) % SIZE_3(gradVertical);

        for (int intDepth = 0; intDepth < SIZE_1(input); intDepth += 1) {
            for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1) {
                dVertical += VALUE_4(gradOutput, intSample, intDepth, intY, intX)
                             * VALUE_4(input, intSample, intDepth, intY + intFilterY, intX + intFilterX) 
                             * VALUE_4(horizontal, intSample, intFilterX, intY, intX);
            }
        }

        gradVertical[intIndex] = dVertical;
    } }
'''

kernel_Sepconv_updateGradHorizontal = '''
    extern "C" __global__ void kernel_Sepconv_updateGradHorizontal(
        const int n,
        const float* input,
        const float* vertical,
        const float* gradOutput,
        float* gradHorizontal
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float dHorizontal = 0.0;

        const int intSample 	= ( intIndex / SIZE_3(gradHorizontal) / SIZE_2(gradHorizontal) / SIZE_1(gradHorizontal) ) % SIZE_0(gradHorizontal);
        const int intFilterX  	= ( intIndex / SIZE_3(gradHorizontal) / SIZE_2(gradHorizontal)                  		) % SIZE_1(gradHorizontal);
        const int intY      	= ( intIndex / SIZE_3(gradHorizontal)                                   				) % SIZE_2(gradHorizontal);
        const int intX      	= ( intIndex                                                    						) % SIZE_3(gradHorizontal);

        for (int intDepth = 0; intDepth < SIZE_1(input); intDepth += 1) {
            for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1) {
                dHorizontal += VALUE_4(gradOutput, intSample, intDepth, intY, intX)
                               * VALUE_4(input, intSample, intDepth, intY + intFilterY, intX + intFilterX) 
                               * VALUE_4(vertical, intSample, intFilterY, intY, intX);
            }
        }

        gradHorizontal[intIndex] = dHorizontal;
    } }
'''

def cupy_kernel(strFunction, objectVariables):
    strKernel = globals()[strFunction]
    # print('strFunction:\n', strFunction)
    # print('')
    # print('')
    # print('strKernel:\n', strKernel)

    while True:
        objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)
        # print('\n\nobjectMatch:\n', objectMatch)

        if objectMatch is None:
            break
        # end

        intArg = int(objectMatch.group(2))
        # print('intArg:\n', intArg)

        strTensor = objectMatch.group(4)
        # print('strTensor:\n', strTensor)
        intSizes = objectVariables[strTensor].size()

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
        # print('strKernel:\n', strKernel)
    # end

    while True:
        objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)
        # print('\n\nobjectMatch:\n', objectMatch)

        if objectMatch is None:
            break
        # end

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')
        # print('intArgs:\n', intArgs)
        # print('strArgs:\n', strArgs)

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

        strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
        # print('strKernel:\n', strKernel)
    # end

    return strKernel
# end

@cupy.util.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
# end

class FunctionSepconv(torch.autograd.Function):
    # def __init__(self):
    #     super(FunctionSepconv, self).__init__()
    # end

    @staticmethod
    def forward(self, input, vertical, horizontal):
        #print("in function forward")
        self.save_for_backward(input, vertical, horizontal)
        #print(input.size(), vertical.size(), horizontal.size())

        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = min(vertical.size(1), horizontal.size(1))
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))

        assert(intInputHeight - intFilterSize == intOutputHeight - 1)
        assert(intInputWidth - intFilterSize == intOutputWidth - 1)

        assert(input.is_contiguous() == True)
        assert(vertical.is_contiguous() == True)
        assert(horizontal.is_contiguous() == True)

        output = input.new_zeros(intSample, intInputDepth, intOutputHeight, intOutputWidth)

        if input.is_cuda == True:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream
            # end

            n = output.nelement()
            cupy_launch('kernel_Sepconv_updateOutput', cupy_kernel('kernel_Sepconv_updateOutput', {
                'input': input,
                'vertical': vertical,
                'horizontal': horizontal,
                'output': output
            }))(
                grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
                block=tuple([ 512, 1, 1 ]),
                args=[ n, input.data_ptr(), vertical.data_ptr(), horizontal.data_ptr(), output.data_ptr() ],
                stream=Stream
            )

        elif input.is_cuda == False:
            raise NotImplementedError()

        # end

        return output
    # end

    @staticmethod
    def backward(self, gradOutput):
        #print("In function backward")
        input, vertical, horizontal = self.saved_tensors

        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = min(vertical.size(1), horizontal.size(1))
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))

        assert(intInputHeight - intFilterSize == intOutputHeight - 1)
        assert(intInputWidth - intFilterSize == intOutputWidth - 1)

        assert(gradOutput.is_contiguous() == True)

        gradInput = input.new_zeros(intSample, intInputDepth, intInputHeight, intInputWidth) if self.needs_input_grad[0] == True else None
        gradVertical = input.new_zeros(intSample, intFilterSize, intOutputHeight, intOutputWidth) if self.needs_input_grad[1] == True else None
        gradHorizontal = input.new_zeros(intSample, intFilterSize, intOutputHeight, intOutputWidth) if self.needs_input_grad[2] == True else None

        if input.is_cuda == True:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream
            # end
            
            if self.needs_input_grad[0]:
                ni = gradInput.nelement()
                cupy_launch('kernel_Sepconv_updateGradInput', cupy_kernel('kernel_Sepconv_updateGradInput', {
                    'input': input,
                    'vertical': vertical,
                    'horizontal': horizontal,
                    'gradOutput': gradOutput,
                    'gradInput': gradInput
                }))(
                    grid=tuple([int((ni + 512 - 1) / 512), 1, 1]),
                    block=tuple([512, 1, 1]),
                    args=[ni, vertical.data_ptr(), horizontal.data_ptr(), gradOutput.data_ptr(), gradInput.data_ptr()],
                    stream=Stream
                )

            if self.needs_input_grad[1]:
                nv = gradVertical.nelement()
                cupy_launch('kernel_Sepconv_updateGradVertical', cupy_kernel('kernel_Sepconv_updateGradVertical', {
                    'input': input,
                    'vertical': vertical,
                    'horizontal': horizontal,
                    'gradOutput': gradOutput,
                    'gradVertical': gradVertical
                }))(
                    grid=tuple([int((nv + 512 - 1) / 512), 1, 1]),
                    block=tuple([512, 1, 1]),
                    args=[nv, input.data_ptr(), horizontal.data_ptr(), gradOutput.data_ptr(), gradVertical.data_ptr()],
                    stream=Stream
                )

            if self.needs_input_grad[2]:
                nh = gradHorizontal.nelement()
                cupy_launch('kernel_Sepconv_updateGradHorizontal', cupy_kernel('kernel_Sepconv_updateGradHorizontal', {
                    'input': input,
                    'vertical': vertical,
                    'horizontal': horizontal,
                    'gradOutput': gradOutput,
                    'gradHorizontal': gradHorizontal
                }))(
                    grid=tuple([int((nh + 512 - 1) / 512), 1, 1]),
                    block=tuple([512, 1, 1]),
                    args=[nh, input.data_ptr(), vertical.data_ptr(), gradOutput.data_ptr(), gradHorizontal.data_ptr()],
                    stream=Stream
                )

        elif input.is_cuda == False:
            raise NotImplementedError()

        # end

        return gradInput, gradVertical, gradHorizontal
    # end
# end

class ModuleSepconv(torch.nn.Module):
    def __init__(self):
        super(ModuleSepconv, self).__init__()
    # end

    def forward(self, tensorFirst, tensorSecond):
        return FunctionSepconv()(tensorFirst, tensorSecond)
    # end
# end
