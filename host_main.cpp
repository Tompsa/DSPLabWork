#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <string>

#include <CL/cl.h>
#include <vector>

#include <cuda_profiler_api.h>

using namespace std;

/* OpenCL macros */ 

#define MAX_ERROR_VALUE 64
#define PREFERRED_PLATFORM "NVIDIA" // Change the platform you want to use here. I.e. Intel
#define PREFERRED_DEVICE CL_DEVICE_TYPE_GPU
#define CL_KERNEL_SOURCE_FILE "kernels/depth_estimator_simple.cl"

/* Simple function for rounding up global work sizes */

int roundUp2(int groupSize, int globalSize) {
	int r = globalSize % groupSize;

	if(r == 0)
		return globalSize;
	else
		return globalSize + groupSize - r;
}

/* A function to check for error code as per cl_int returned by OpenCl
Parameter errCheck Error value as cl_int
Parameter msg User provided error message
Return True if Error found, False otherwise */

int cl_errCheck(const cl_int errCheck, const char * msg, bool exitOnError)
{
	char *cl_error[MAX_ERROR_VALUE] = {
    "CL_SUCCESS",                         // 0
    "CL_DEVICE_NOT_FOUND",                //-1
    "CL_DEVICE_NOT_AVAILABLE",            //-2
    "CL_COMPILER_NOT_AVAILABLE",          //-3
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",   //-4
    "CL_OUT_OF_RESOURCES",                //-5
    "CL_OUT_OF_HOST_MEMORY",              //-6
    "CL_PROFILING_INFO_NOT_AVAILABLE",    //-7
    "CL_MEM_COPY_OVERLAP",                //-8
    "CL_IMAGE_FORMAT_MISMATCH",           //-9
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",      //-10
    "CL_BUILD_PROGRAM_FAILURE",           //-11
    "CL_MAP_FAILURE",                     //-12
    "",                                   //-13
    "",                                   //-14
    "",                                   //-15
    "",                                   //-16
    "",                                   //-17
    "",                                   //-18
    "",                                   //-19
    "",                                   //-20
    "",                                   //-21
    "",                                   //-22
    "",                                   //-23
    "",                                   //-24
    "",                                   //-25
    "",                                   //-26
    "",                                   //-27
    "",                                   //-28
    "",                                   //-29
    "CL_INVALID_VALUE",                   //-30
    "CL_INVALID_DEVICE_TYPE",             //-31
    "CL_INVALID_PLATFORM",                //-32
    "CL_INVALID_DEVICE",                  //-33
    "CL_INVALID_CONTEXT",                 //-34
    "CL_INVALID_QUEUE_PROPERTIES",        //-35
    "CL_INVALID_COMMAND_QUEUE",           //-36
    "CL_INVALID_HOST_PTR",                //-37
    "CL_INVALID_MEM_OBJECT",              //-38
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", //-39
    "CL_INVALID_IMAGE_SIZE",              //-40
    "CL_INVALID_SAMPLER",                 //-41
    "CL_INVALID_BINARY",                  //-42
    "CL_INVALID_BUILD_OPTIONS",           //-43
    "CL_INVALID_PROGRAM",                 //-44
    "CL_INVALID_PROGRAM_EXECUTABLE",      //-45
    "CL_INVALID_KERNEL_NAME",             //-46
    "CL_INVALID_KERNEL_DEFINITION",       //-47
    "CL_INVALID_KERNEL",                  //-48
    "CL_INVALID_ARG_INDEX",               //-49
    "CL_INVALID_ARG_VALUE",               //-50
    "CL_INVALID_ARG_SIZE",                //-51
    "CL_INVALID_KERNEL_ARGS",             //-52
    "CL_INVALID_WORK_DIMENSION ",         //-53
    "CL_INVALID_WORK_GROUP_SIZE",         //-54
    "CL_INVALID_WORK_ITEM_SIZE",          //-55
    "CL_INVALID_GLOBAL_OFFSET",           //-56
    "CL_INVALID_EVENT_WAIT_LIST",         //-57
    "CL_INVALID_EVENT",                   //-58
    "CL_INVALID_OPERATION",               //-59
    "CL_INVALID_GL_OBJECT",               //-60
    "CL_INVALID_BUFFER_SIZE",             //-61
    "CL_INVALID_MIP_LEVEL",               //-62
    "CL_INVALID_GLOBAL_WORK_SIZE"};       //-63


    if(errCheck != CL_SUCCESS) {
        printf("OpenCL Error: %d %s %s\n", errCheck, (char *)(cl_error[-errCheck]), msg);

        if(exitOnError) {
            exit(-1);
        }
        return true;
    }
    return false;
}

/* Input and output image declaration */

char * defLeftInputImageName = "./images/view1.pgm";
char * defRightInputImageName = "./images/view5.pgm";
char * defOutputImageName = "./images/depth_map.pgm";
char * defOutputImageName2 = "./images/right_depth.pgm";

/* Portable graymap (PGM) functions */
#ifdef _WIN32 | _WIN64
	extern "C" {
	int ReadPGMHeader(char * filename, int * imwidth, int* imheight);
	int WritePGM(char *filename, unsigned char *source, unsigned int width, unsigned int height);
	int ReadPGMData(char *filename, unsigned char * target, int width, int height, int dataoffset);
	int AsciiToInt(char *asc);
	}
#else
	int ReadPGMHeader(char * filename, int * imwidth, int* imheight);
	int WritePGM(char *filename, unsigned char *source, unsigned int width, unsigned int height);
	int ReadPGMData(char *filename, unsigned char * target, int width, int height, int dataoffset);
	int AsciiToInt(char *asc);
#endif


/* OpenCL objects declaration */
struct OpenCLObjects
{
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel[2];
};
OpenCLObjects openCLObjects;

/////////////////////////////////// Main function  //////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
	int imWidthL, imHeightL, imWidthR, imHeightR, imSize, width, height, size;
	int channels = 1;
	int headerlengthL, headerlengthR;
	unsigned char *imgL = NULL, *imgR = NULL, *resultL = NULL, *resultR = NULL;
	char *imNameL = NULL, *imNameR = NULL;

	if (argc < 3)
	{
		imNameL = defLeftInputImageName;
		imNameR = defRightInputImageName;
	}
	else
	{
		imNameL = argv[1];
		imNameR = argv[2];
	}

	/* Image information check*/
	headerlengthL = ReadPGMHeader(imNameL, &imWidthL, &imHeightL);
	headerlengthR = ReadPGMHeader(imNameR, &imWidthR, &imHeightR);

	if (imWidthL != imWidthR || imHeightL != imHeightR)
	{
		fprintf(stderr, "Images are not the same size!\n");
		exit(1);
	}

	width = imWidthL;
	height = imHeightL;
	imSize = width * height;

	imgL = (unsigned char*)malloc(sizeof(unsigned char)*imSize);
	imgR = (unsigned char*)malloc(sizeof(unsigned char)*imSize);
	resultL = (unsigned char*)malloc(sizeof(unsigned char)*imSize); // result depth map
	resultR = (unsigned char*)malloc(sizeof(unsigned char)*imSize);

	if (imgL == NULL || imgR == NULL)
	{
		fprintf(stderr, "Memory allocation error!\n");
		exit(1);
	}

	ReadPGMData(imNameL, imgL, width, height, headerlengthL);
	ReadPGMData(imNameR, imgR, width, height, headerlengthR);

	int WIN_SIZE = 8;
	int MAX_DISP = 90;

	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	/*
	OPENCLinitialization (Platform, Device, program, etc.)
	*/
	///////////////////////////////////////////////////////////////////////////////////////////////////////////
	cudaProfilerStart();
	unsigned int idP = -1;
	cl_uint num_platforms = 0, numDevices = 0, i = 0;
	cl_int errCheck = 0;
	size_t platform_name_length = 0, workitem_size[3], workgroup_size, address_bits;

	/* Step 1 */
	/*	Query for all available OpenCL platforms and devices on the system.
		Select a platform that has the required PREFERRED_PLATFORM substring using strstr-function.
		*/

	// Get total number of the available platforms.
	errCheck = clGetPlatformIDs(0, 0, &num_platforms);
	cl_errCheck(errCheck, "Platform inquiry", true);
	printf("Number of available platforms: %u \n", num_platforms);

	// Get IDs for all platforms.
	vector<cl_platform_id> platforms(num_platforms);
	errCheck = clGetPlatformIDs(num_platforms, &platforms[0], 0);
	cl_errCheck(errCheck, "clGetPlatformIds", true);

	for (i = 0; i < num_platforms; i++)
	{

		// Get the size of the platform name in bytes.
		errCheck = clGetPlatformInfo(
			platforms[i],
			CL_PLATFORM_NAME,
			0,
			0,
			&platform_name_length
			);
		cl_errCheck(errCheck, "clGetPlatformInfo", true);

		// Get the actual name for the i-th platform.
		vector<char> platform_name(platform_name_length);
		errCheck = clGetPlatformInfo(
			platforms[i],
			CL_PLATFORM_NAME,
			platform_name_length,
			&platform_name[0],
			0
			);
		cl_errCheck(errCheck, "clGetPlatformIInfo Names", true);

		//Print out the platform id and name
		string platformName = &platform_name[0];
		printf("\n[%u] %s \n", i, (platformName).c_str());

		// Check if the platform is the preferred platform
		if (strstr(&platform_name[0], PREFERRED_PLATFORM))
		{
			openCLObjects.platform = platforms[i];
			idP = i;
		}
	}

	if (idP == -1) {
		printf("Preferred platform not found. Exiting...");
		exit(1);
	}
	else
		printf("\nPlaform ID [%u] selected \n", idP);

	/////////////// This is where the exercise work begins //////////////////////////////////////////////////////////
	/* STEP 2 */
	/* All the platforms are now queried and selected. Next you need to query all the available devices for the platforms.
	Of course you can only query the devices for the selected platform and then select a suitable device.
	Depending on your approach place the device query inside or outside of the above for loop. The reason for scanning all
	the devices in each platform is just to show you what device options you might have.
	*/

	cl_device_id device;
	clGetDeviceIDs(platforms[idP], CL_DEVICE_TYPE_GPU, 1, &device, NULL);

	/* STEP 3 */
	/* You have now selected a platform and a device to use either a CPU or a GPU in our case. In this exercise we simply use one device at a
	time. Of course you are free to implement a multidevice setup if you want. Now you need to create context for the selected device.
	*/

	cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &errCheck);

	/* STEP 4. */
	/* Query for the OpenCL device that was used for context creation using clGetContextInfo. This step is just to check
		that no errors occured during context creation step. Error handling is very important in OpenCL since the bug might be in the host or
		kernel code. Use the errCheck-function on every step to indicate the location of the possible bug.
		*/

	size_t context_length = 0;
	errCheck = clGetContextInfo(
		context,
		CL_CONTEXT_DEVICES,
		0,
		0,
		&context_length
		);
	cl_errCheck(errCheck, "clCreateContext", true);

	/* STEP 5. */
	/*	Create OpenCL program from the kernel file source code
		First read the source kernel code in from the .cl file as an array of char's.
		Then "submit" the source code of the kernel to OpenCL and create a program object with it.
		*/

	FILE *f;
	size_t source_size;
	char *source_str;

	#pragma warning (disable : 4996)
	f = fopen(CL_KERNEL_SOURCE_FILE, "r");
	if (!f) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(0x10000);
	source_size = fread(source_str, 1, 0x10000, f);
	fclose(f);

	cl_program program = clCreateProgramWithSource(context, 1,
		(const char**)&source_str, (const size_t*)&source_size, &errCheck);
	cl_errCheck(errCheck, "clCreateProgramWithSource", true);


	/* STEP 6. */
	/* Build the program. The program is now created but not built. Next you need to build it
	*/

	errCheck = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	size_t log_size;
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	char *log = (char *)malloc(log_size);

	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
	printf("%s", log);

	cl_errCheck(errCheck, "clBuildProgram", true);

	/* STEP 7. */
	/*	Extract the kernel/kernels from the built program. The program consists of one or more kernels. Each kernel needs to be enqueued for
		execution from the host code. Creating a kernel via clCreateKernel is similar to obtaining an entry point of a specific function
		in an OpenCL program.
		*/

	cl_kernel kernel = clCreateKernel(program, "depth_estimator_simple", &errCheck);
	cl_errCheck(errCheck, "clCreateKernel", true);
	cl_kernel kernel_r = clCreateKernel(program, "depth_estimator_simple_r", &errCheck);
	cl_errCheck(errCheck, "clCreateKernel", true);
	cl_kernel kernel_cc = clCreateKernel(program, "cross_check", &errCheck);
	cl_errCheck(errCheck, "clCreateKernel", true);
	cl_kernel kernel_of = clCreateKernel(program, "occlusion_filling", &errCheck);
	cl_errCheck(errCheck, "clCreateKernel", true);

	/* STEP 8. */
	/*	Now that you have created the kernel/kernels, you also need to enqueue them for execution to the selected device. The command queu can
	be either in-order or out-of-order type depending on your approach. In-order command queue is a good starting point.
	*/

	cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &errCheck);
	cl_errCheck(errCheck, "clCreateCommandQueue", true);

	/* STEP 9. */
	/* Allocate device memory. You need to at least allocate memory on the device for the input and output images.
	Remember the error handling for the memory objects also.
	*/

	cl_mem image_L = clCreateBuffer(context, CL_MEM_READ_ONLY,
		imSize * sizeof(char), NULL, &errCheck);
	cl_mem image_R = clCreateBuffer(context, CL_MEM_READ_WRITE,
		imSize * sizeof(char), NULL, &errCheck);
	cl_mem result_image = clCreateBuffer(context, CL_MEM_READ_WRITE,
		imSize * sizeof(char), NULL, &errCheck);
	cl_mem result_image_R = clCreateBuffer(context, CL_MEM_READ_WRITE,
		imSize * sizeof(char), NULL, &errCheck);

	cl_errCheck(errCheck, "clCreateBuffer", true);

	/* STEP 10. */
	// Enqueue the memory objects for writing to device using clEnqueueWriteBuffer/clEnqueueWriteImage.

	size_t local_item_size[2] = { 32, 32 };
	size_t global_item_size[2] = { roundUp2(32, width), roundUp2(32, height) };
	int local_padded_width = 64;
	int local_padded_height = 64;
	int local_padded_width_r = 128;
	int local_padded_height_r = 64;
	size_t local_padded_size = local_padded_width * local_padded_height * sizeof(char);
	size_t local_padded_size_r = local_padded_width_r * local_padded_height_r * sizeof(char);


	errCheck = clEnqueueWriteBuffer(queue, image_L, CL_TRUE, 0, imSize * sizeof(char), imgL, 0, NULL, NULL);
	errCheck = clEnqueueWriteBuffer(queue, image_R, CL_TRUE, 0, imSize * sizeof(char), imgR, 0, NULL, NULL);
	errCheck = clEnqueueWriteBuffer(queue, result_image, CL_TRUE, 0, imSize * sizeof(char), resultL, 0, NULL, NULL);
	errCheck = clEnqueueWriteBuffer(queue, result_image_R, CL_TRUE, 0, imSize * sizeof(char), resultR, 0, NULL, NULL);
	cl_errCheck(errCheck, "clEnqueueWriteBuffer", true);

	/* STEP 11. */
	/*	Set the kernel arguments. Input images/buffers, output image/buffer, etc. Also set the global and local workgroup sizes if you have not
	done it yet. Remember that the global work group size needs to be a multiple of the local workgroup size.
	You can use the RoundUp2-function to make sure that this condition is met.
	*/


	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&image_L);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&image_R);
	clSetKernelArg(kernel, 2, sizeof(int), (void *)&width);
	clSetKernelArg(kernel, 3, sizeof(int), (void *)&height);
	clSetKernelArg(kernel, 4, sizeof(int), (void *)&WIN_SIZE);
	clSetKernelArg(kernel, 5, sizeof(int), (void *)&MAX_DISP);
	clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&result_image);
	clSetKernelArg(kernel, 7, local_padded_size, NULL);
	clSetKernelArg(kernel, 8, local_padded_size_r, NULL);
	clSetKernelArg(kernel, 9, sizeof(int), (void*)&local_padded_width);
	clSetKernelArg(kernel, 10, sizeof(int), (void*)&local_padded_height);
	clSetKernelArg(kernel, 11, sizeof(int), (void*)&local_padded_width_r);


	/* STEP 12. */
	/*	Queue the kernel for execution.
		If you have more than one kernel, repeat step 11. and 12. for all of them.
	*/

	cl_event event, event2, event3, event4;
	cl_ulong time_start, time_end;
	double total_time = 0;

	errCheck = clEnqueueNDRangeKernel(queue, kernel, 2,
		NULL,
		global_item_size,
		local_item_size,
		0, NULL, &event);
	cl_errCheck(errCheck, "clEnqueueNDRangeKernel", true);

	errCheck = clEnqueueReadBuffer(queue, result_image, CL_TRUE, 0, imSize * sizeof(char), resultL, 0, NULL, NULL);

	cl_errCheck(errCheck, "clEnqueueReadBuffer", true);

	clWaitForEvents(1, &event);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time += time_end - time_start;

	printf("\nSSD execution time in milliseconds = %0.3f ms\n", ((time_end - time_start) / 1000000.0));
	
	// Right image depth map
	
	clSetKernelArg(kernel_r, 0, sizeof(cl_mem), (void *)&image_L);
	clSetKernelArg(kernel_r, 1, sizeof(cl_mem), (void *)&image_R);
	clSetKernelArg(kernel_r, 2, sizeof(int), (void *)&width);
	clSetKernelArg(kernel_r, 3, sizeof(int), (void *)&height);
	clSetKernelArg(kernel_r, 4, sizeof(int), (void *)&WIN_SIZE);
	clSetKernelArg(kernel_r, 5, sizeof(int), (void *)&MAX_DISP);
	clSetKernelArg(kernel_r, 6, sizeof(cl_mem), (void *)&result_image_R);
	clSetKernelArg(kernel_r, 7, local_padded_size, NULL);
	clSetKernelArg(kernel_r, 8, local_padded_size_r, NULL);
	clSetKernelArg(kernel_r, 9, sizeof(int), (void*)&local_padded_width);
	clSetKernelArg(kernel_r, 10, sizeof(int), (void*)&local_padded_height);
	clSetKernelArg(kernel_r, 11, sizeof(int), (void*)&local_padded_width_r);

	errCheck = clEnqueueNDRangeKernel(queue, kernel_r, 2,
		NULL,
		global_item_size,
		local_item_size,
		0, NULL, &event4);
	cl_errCheck(errCheck, "clEnqueueNDRangeKernel", true);

	errCheck = clEnqueueReadBuffer(queue, result_image_R, CL_TRUE, 0, imSize * sizeof(char), resultR, 0, NULL, NULL);
	cl_errCheck(errCheck, "clEnqueueReadBuffer", true);
	
	clWaitForEvents(1, &event4);
	clGetEventProfilingInfo(event4, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event4, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time += time_end - time_start;

	printf("\nRight image SSD time in milliseconds = %0.3f ms\n", ((time_end - time_start) / 1000000.0));

	// Cross_check

	int threshold = 20;

	clSetKernelArg(kernel_cc, 0, sizeof(cl_mem), (void *)&result_image);
	clSetKernelArg(kernel_cc, 1, sizeof(cl_mem), (void *)&result_image_R);
	clSetKernelArg(kernel_cc, 2, sizeof(int), (void *)&threshold);
	clSetKernelArg(kernel_cc, 3, sizeof(int), (void *)&width);

	errCheck = clEnqueueNDRangeKernel(queue, kernel_cc, 2,
		NULL,
		global_item_size,
		local_item_size,
		0, NULL, &event2);
	cl_errCheck(errCheck, "clEnqueueNDRangeKernel", true);

	errCheck = clEnqueueReadBuffer(queue, result_image, CL_TRUE, 0, imSize * sizeof(char), resultL, 0, NULL, NULL);
	cl_errCheck(errCheck, "clEnqueueReadBuffer", true);
	
	clWaitForEvents(1, &event2);
	clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event2, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time += time_end - time_start;

	printf("\nCross check execution time in milliseconds = %0.3f ms\n", ((time_end - time_start) / 1000000.0));
	
	// Occlusion_filling
	
	clSetKernelArg(kernel_of, 0, sizeof(cl_mem), (void *)&result_image);
	clSetKernelArg(kernel_of, 1, sizeof(int), (void *)&width);
	clSetKernelArg(kernel_of, 2, sizeof(int), (void *)&height);

	errCheck = clEnqueueNDRangeKernel(queue, kernel_of, 2,
		NULL,
		global_item_size,
		local_item_size,
		0, NULL, &event3);
	cl_errCheck(errCheck, "clEnqueueNDRangeKernel", true);
	
	/* STEP 13. */
	/* Read the output buffer back to the host using clEnqueueReadBuffer/clEnqueueReadImage */

	errCheck = clEnqueueReadBuffer(queue, result_image, CL_TRUE, 0, imSize * sizeof(char), resultL, 0, NULL, NULL);
	cl_errCheck(errCheck, "clEnqueueReadBuffer", true); 

	clWaitForEvents(1, &event3);
	clGetEventProfilingInfo(event3, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(event3, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time += time_end - time_start;

	printf("\nOccl Fill execution time in milliseconds = %0.3f ms\n", ((time_end - time_start) / 1000000.0));

	printf("\nTotal execution time in milliseconds = %0.3f ms\n", (total_time / 1000000.0));
	
	/* Write the result image to a pgm file */
	WritePGM( defOutputImageName, resultL, width, height);
	WritePGM(defOutputImageName2, resultR, width, height);

	/* STEP 14 */
	/* Perform "cleanup". Release all OpenCL objects i.e. kernels, program, memory */

	errCheck = clFlush(queue);
	errCheck = clFinish(queue);
	errCheck = clReleaseCommandQueue(queue);
	errCheck = clReleaseKernel(kernel);
	errCheck = clReleaseKernel(kernel_r);
	errCheck = clReleaseKernel(kernel_cc);
	errCheck = clReleaseKernel(kernel_of);
	errCheck = clReleaseMemObject(image_R);
	errCheck = clReleaseMemObject(image_L);
	errCheck = clReleaseMemObject(result_image);
	errCheck = clReleaseMemObject(result_image_R);
	errCheck = clReleaseProgram(program);
	errCheck = clReleaseContext(context);
	free(imgL);
	free(imgR);
	free(resultL);
	free(resultR);
	cudaProfilerStop();
	return 0;
}


