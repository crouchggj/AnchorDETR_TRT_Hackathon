#!/usr/bin/python

import os
import sys
import ctypes
import argparse
import numpy as np
from glob import glob
from time import time_ns
from datetime import datetime as dt
from cuda import cudart
import tensorrt as trt
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

tableHead = \
    """
lt: Latency (ms)
a0: maximum of absolute difference of output 0
r0: median of relative difference of output 0
a1: maximum of absolute difference of output 1
r1: median of relative difference of output 1
--------+--------+--------+---------+---------+---------+---------+--------------
    name|      lt|     fps|       a0|       r0|       a1|       r1| output check
--------+--------+--------+---------+---------+---------+---------+--------------
"""


def check(a, b, weak=False, epsilon=1e-5):
    if weak:
        res = np.all(np.abs(a - b) < epsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.median(np.abs(a - b) / (np.abs(b) + epsilon))
    return res, diff0, diff1


def run(plan=ROOT):
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')

    # load custom plugin
    planFilePath = os.path.join(ROOT, "build")
    soFileList = glob(planFilePath + "/*.so")
    if len(soFileList) > 0:
        print("Find Plugin %s!" % soFileList)
    else:
        print("No Plugin!")
    for soFile in soFileList:
        ctypes.cdll.LoadLibrary(soFile)

    plan = plan[0]
    if os.path.isfile(plan):
        with open(plan, 'rb') as encoderF:
            engine = trt.Runtime(
                logger).deserialize_cuda_engine(encoderF.read())
        if engine is None:
            print("Failed loading %s" % plan)
            exit()
        print("Succeeded loading %s" % plan)
    else:
        print("Not find plan in %s" % (plan))
        exit(0)

    nInput = np.sum([engine.binding_is_input(i)
                    for i in range(engine.num_bindings)])
    nOutput = engine.num_bindings - nInput
    context = engine.create_execution_context()

    testFileList = glob("./data/*.npz")
    print(tableHead)
    for testFile in testFileList:
        ioData = np.load(testFile)
        testData = ioData["inputs"]
        # print(testData.shape)
        # context.set_binding_shape(0, shape)

        bufferH = []
        bufferH.append(testData.astype(np.float32).reshape(-1))
        for i in range(nInput, nInput + nOutput):
            bufferH.append(np.empty(context.get_binding_shape(i),
                                    dtype=trt.nptype(engine.get_binding_dtype(i))))

        bufferD = []
        for i in range(nInput + nOutput):
            bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

        for i in range(nInput):
            cudart.cudaMemcpy(bufferD[i],
                              bufferH[i].ctypes.data,
                              bufferH[i].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        context.execute_v2(bufferD)

        for i in range(nInput, nInput + nOutput):
            cudart.cudaMemcpy(bufferH[i].ctypes.data,
                              bufferD[i],
                              bufferH[i].nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

        # warm up
        for i in range(10):
            context.execute_v2(bufferD)

        # test infernece time
        t0 = time_ns()
        for i in range(30):
            context.execute_v2(bufferD)
        t1 = time_ns()
        timePerInference = (t1-t0)/1000/1000/30

        indexLogitsOut = engine.get_binding_index('pred_logits')
        indexBoxesOut = engine.get_binding_index('pred_boxes')

        check0 = check(bufferH[indexLogitsOut],
                       ioData['pred_logits'], True)
        check1 = check(bufferH[indexBoxesOut],
                       ioData['pred_boxes'], True)
        string = "%8s,%8.3f,%8.2f,%9.3e,%9.3e,%9.3e,%9.3e,   %s" % (os.path.basename(testFile).split('_')[0],
                                                                    timePerInference,
                                                                    1000 / timePerInference,
                                                                    check0[1],
                                                                    check0[2],
                                                                    check1[1],
                                                                    check1[2],
                                                                    "Good" if check0[1] < 3.5e-2 and check0[2] < 2e-3 and check1[2] < 1e-1 else "Bad")
        print(string)
        # print("Case [%s] lantecy: %8.3f ms  diff_logits: %9.3e diff_boxes: %9.3e %s" %
        #       (testFile, timePerInference,
        #        check0[1], check1[1],
        #        "PASS" if check0[1] < 1e-2 and check0[2] < 1e-2 and check1[2] < 1e-2 else "NOT PASS"))

        for i in range(nInput + nOutput):
            cudart.cudaFree(bufferD[i])


def main(opt):
    run(**vars(opt))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', nargs='+', type=str,
                        default=ROOT / 'model.plan', help='model.plan path(s)')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
