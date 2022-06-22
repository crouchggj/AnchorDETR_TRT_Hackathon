import onnx_graphsurgeon as gs
import numpy as np
import onnx
import sys

bRemoveOps = True
bFoldConstant = True
bLayerNormPlugin = True
nLayerNormPlugin = 0

onnxPath = str(sys.argv[1])
newModelPath = str(sys.argv[2])
graph = gs.import_onnx(onnx.load(onnxPath))

tmap = graph.tensors()

# remove useless op
if bRemoveOps:
    conv107_node = [node for node in graph.nodes if node.name == "Conv_107"][0]
    split_node = [node for node in graph.nodes if node.name == "Split_30"][0]
    conv107_node.inputs[0] = split_node.inputs[0]
    for node in graph.nodes:
        if node.op == 'Expand':
            if node.o().op == 'Reshape':
                print(node.o().name)
                print(node.o().outputs)
                node.o().o().inputs[2] = node.inputs[0]
                node.o().outputs = []
            elif node.o().op == 'Tile':
                node.o().o().inputs[0] = node.inputs[0]
                node.o().outputs = []
            else:
                node.o().inputs[0] = node.inputs[0]
                node.outputs = []

# fold constant
if bFoldConstant:
    for node in graph.nodes:
        if node.op == 'Log' and \
                node.i().op == 'Div':
            input0 = np.clip(node.i().i(0).inputs[0].values,
                             node.i().i(0).inputs[1].values,
                             np.finfo(np.float32).max)
            input1 = np.clip(node.i().i(1).inputs[0].values,
                             node.i().i(1).inputs[1].values,
                             np.finfo(np.float32).max)
            output = np.log(input0 / input1)
            constantData = gs.Constant("Data", np.ascontiguousarray(output))
            node.o().inputs[1] = constantData
            node.outputs = []

# fuse layerNorm
if bLayerNormPlugin:
    for node in graph.nodes:
        if node.op == 'ReduceMean' and \
                node.o().op == 'Sub' and node.o().inputs[0] == node.inputs[0] and \
                node.o().o(0).op == 'Pow' and node.o().o(1).op == 'Div' and \
                node.o().o(0).o().op == 'ReduceMean' and \
                node.o().o(0).o().o().op == 'Add' and \
                node.o().o(0).o().o().o().op == 'Sqrt' and \
                node.o().o(0).o().o().o().o().op == 'Div' and node.o().o(0).o().o().o().o() == node.o().o(1) and \
                node.o().o(1).o().op == 'Mul' and \
                node.o().o(1).o().o().op == 'Add':

            inputTensor = node.inputs[0]
            lastDivNode = node.o().o(0).o().o().o().o().o().o()

            layerNormN = gs.Node("LayerNorm", "LayerNormN-" + str(nLayerNormPlugin),
                                 inputs=[inputTensor, node.o().o(1).o().inputs[1], node.o().o(1).o().o().inputs[1]], outputs=[lastDivNode.outputs[0]])
            graph.nodes.append(layerNormN)
            nLayerNormPlugin += 1

            lastDivNode.outputs = []
            continue


# Remove the now-dangling subgraph.
graph.cleanup().toposort()

print("nLayerNormPlugin: %d" % (nLayerNormPlugin))
onnx.save(gs.export_onnx(graph), newModelPath)
