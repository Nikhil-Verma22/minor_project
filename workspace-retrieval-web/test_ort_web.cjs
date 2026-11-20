const ort = require('onnxruntime-web');
const fs = require('fs');
const path = require('path');

async function testOrt() {
    console.log('Testing onnxruntime-web in Node...');
    console.log('ORT Web version:', ort.version);
    
    const modelPath = path.resolve('../checkpoints/PSG/react++_yolo12n/react_pp_yolo12n.onnx');
    console.log('Model path:', modelPath, 'exists:', fs.existsSync(modelPath));
    
    try {
        const modelBuffer = fs.readFileSync(modelPath);
        console.log('Read model buffer, bytes:', modelBuffer.byteLength);
        
        console.log('Creating InferenceSession with WASM...');
        const session = await ort.InferenceSession.create(modelBuffer.buffer, {
            executionProviders: ['wasm']
        });
        console.log('Session created successfully!');
        console.log('Inputs:', session.inputNames);
        console.log('Outputs:', session.outputNames);
        
        // Create dummy input [1, 3, 640, 640]
        const inputData = new Float32Array(1 * 3 * 640 * 640);
        const inputTensor = new ort.Tensor('float32', inputData, [1, 3, 640, 640]);
        
        console.log('Running dummy forward pass...');
        const t0 = Date.now();
        const results = await session.run({ [session.inputNames[0]]: inputTensor });
        const latency = Date.now() - t0;
        console.log(`Forward pass successful in ${latency} ms!`);
        for (const outName of session.outputNames) {
            const outTensor = results[outName];
            console.log(`  Output [${outName}]: dims=[${outTensor.dims}], type=${outTensor.type}, size=${outTensor.data.length}`);
        }
    } catch (e) {
        console.error('Test error:', e);
    }
}

testOrt();
