
const express = require('express');
const faceapi = require('face-api.js');
const canvas = require('canvas');
const fs = require('fs');
const path = require('path');

// Monkey patch for face-api.js to work with canvas
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const app = express();

// Enhanced middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// CORS middleware for cross-origin requests
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');

    if (req.method === 'OPTIONS') {
        return res.sendStatus(200);
    } else {
        next();
    }
});

// Logging middleware
app.use((req, res, next) => {
    const timestamp = new Date().toISOString();
    console.log(`${timestamp} - ${req.method} ${req.path}`);
    next();
});

// Load models
const MODEL_URL = './models';
let modelsLoaded = false;
let loadingProgress = {};

async function loadModels() {
    try {
        console.log('\n Loading face-api.js models...');
        const startTime = Date.now();

        // Define all required models
        const modelLoaders = [
            { name: 'TinyFaceDetector', loader: () => faceapi.nets.tinyFaceDetector.loadFromDisk(MODEL_URL) },
            { name: 'FaceLandmark68Net', loader: () => faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL) },
            { name: 'FaceRecognitionNet', loader: () => faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL) },
            { name: 'FaceExpressionNet', loader: () => faceapi.nets.faceExpressionNet.loadFromDisk(MODEL_URL) },
            { name: 'AgeGenderNet', loader: () => faceapi.nets.ageGenderNet.loadFromDisk(MODEL_URL) },
            { name: 'SsdMobilenetv1', loader: () => faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL) }
        ];

        // Load models with progress tracking
        for (const model of modelLoaders) {
            try {
                console.log(`  Loading ${model.name}...`);
                await model.loader();
                loadingProgress[model.name] = 'loaded';
                console.log(`${model.name} loaded successfully`);
            } catch (error) {
                console.error(`Failed to load ${model.name}:`, error.message);
                loadingProgress[model.name] = 'failed';
                throw error;
            }
        }

        const endTime = Date.now();
        modelsLoaded = true;
        console.log(`\n All models loaded successfully in ${endTime - startTime}ms`);
        return true;
    } catch (error) {
        console.error(' Error loading models:', error);
        modelsLoaded = false;
        return false;
    }
}

// Enhanced health check endpoint
app.get('/health', (req, res) => {
    const memUsage = process.memoryUsage();
    const health = {
        status: 'ok',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        modelsLoaded: modelsLoaded,
        loadingProgress: loadingProgress,
        memory: {
            rss: Math.round(memUsage.rss / 1024 / 1024) + ' MB',
            heapTotal: Math.round(memUsage.heapTotal / 1024 / 1024) + ' MB',
            heapUsed: Math.round(memUsage.heapUsed / 1024 / 1024) + ' MB'
        },
        nodeVersion: process.version
    };

    res.json(health);
});

// Model status endpoint
app.get('/models', (req, res) => {
    res.json({
        modelsLoaded: modelsLoaded,
        loadingProgress: loadingProgress,
        availableModels: Object.keys(loadingProgress)
    });
});

// Enhanced face detection with multiple strategies
app.post('/extract_embedding', async (req, res) => {
    if (!modelsLoaded) {
        return res.status(503).json({ 
            success: false,
            error: 'Models not loaded yet',
            progress: loadingProgress
        });
    }

    const startTime = Date.now();

    try {
        const { image } = req.body;

        if (!image) {
            return res.status(400).json({
                success: false,
                error: 'No image provided in request body'
            });
        }

        // Load image from base64
        let img;
        try {
            const buffer = Buffer.from(image, 'base64');
            img = await canvas.loadImage(buffer);
        } catch (error) {
            return res.status(400).json({
                success: false,
                error: 'Invalid image format or corrupted base64 data'
            });
        }

        // Multiple detection strategies for better success rate
        const detectionStrategies = [
            {
                name: 'TinyFaceDetector_High',
                options: new faceapi.TinyFaceDetectorOptions({ 
                    inputSize: 416, 
                    scoreThreshold: 0.3 
                })
            },
            {
                name: 'TinyFaceDetector_Medium',
                options: new faceapi.TinyFaceDetectorOptions({ 
                    inputSize: 320, 
                    scoreThreshold: 0.4 
                })
            },
            {
                name: 'TinyFaceDetector_Low',
                options: new faceapi.TinyFaceDetectorOptions({ 
                    inputSize: 224, 
                    scoreThreshold: 0.5 
                })
            },
            {
                name: 'SsdMobilenetv1',
                options: new faceapi.SsdMobilenetv1Options({ 
                    minConfidence: 0.3 
                })
            }
        ];

        let detection = null;
        let usedStrategy = null;

        // Try each strategy until one succeeds
        for (const strategy of detectionStrategies) {
            try {
                const detectionStartTime = Date.now();

                detection = await faceapi.detectSingleFace(img, strategy.options)
                    .withFaceLandmarks()
                    .withFaceDescriptor();

                if (detection && detection.descriptor) {
                    usedStrategy = strategy.name;
                    console.log(`Detection successful with ${strategy.name} in ${Date.now() - detectionStartTime}ms`);
                    break;
                }
            } catch (err) {
                console.warn(` Detection failed with ${strategy.name}:`, err.message);
                continue;
            }
        }

        const totalTime = Date.now() - startTime;

        if (detection && detection.descriptor) {
            // Successful detection
            const response = {
                success: true,
                embedding: Array.from(detection.descriptor),
                inferenceTime: totalTime,
                strategy: usedStrategy,
                boundingBox: {
                    x: Math.round(detection.detection.box.x),
                    y: Math.round(detection.detection.box.y),
                    width: Math.round(detection.detection.box.width),
                    height: Math.round(detection.detection.box.height)
                },
                confidence: detection.detection.score || 0.0,
                landmarks: detection.landmarks ? detection.landmarks.positions.length : 0
            };

            console.log(` Embedding extracted successfully (${totalTime}ms)`);
            res.json(response);
        } else {
            // Failed detection
            const response = {
                success: false,
                error: 'No face detected with any strategy',
                inferenceTime: totalTime,
                strategiesTried: detectionStrategies.map(s => s.name)
            };
            console.log(` Face detection failed (${totalTime}ms)`);
            res.json(response);
        }
    } catch (error) {
        const totalTime = Date.now() - startTime;
        console.error(' Extraction error:', error);
        res.status(500).json({ 
            success: false,
            error: error.message,
            inferenceTime: totalTime
        });
    }
});

// Batch processing endpoint with progress tracking
app.post('/batch_extract', async (req, res) => {
    if (!modelsLoaded) {
        return res.status(503).json({ 
            error: 'Models not loaded yet',
            progress: loadingProgress
        });
    }

    try {
        const { images } = req.body;

        if (!images || !Array.isArray(images)) {
            return res.status(400).json({ 
                error: 'Images array is required' 
            });
        }

        console.log(` Starting batch processing of ${images.length} images`);
        const results = [];
        const startTime = Date.now();

        for (let i = 0; i < images.length; i++) {
            const imageStartTime = Date.now();

            try {
                const img = await canvas.loadImage(Buffer.from(images[i], 'base64'));

                const detection = await faceapi.detectSingleFace(img, 
                    new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.3 }))
                    .withFaceLandmarks()
                    .withFaceDescriptor();

                const imageTime = Date.now() - imageStartTime;

                if (detection && detection.descriptor) {
                    results.push({
                        index: i,
                        success: true,
                        embedding: Array.from(detection.descriptor),
                        inferenceTime: imageTime,
                        confidence: detection.detection.score || 0.0
                    });
                } else {
                    results.push({
                        index: i,
                        success: false,
                        inferenceTime: imageTime,
                        error: 'No face detected'
                    });
                }
            } catch (error) {
                results.push({
                    index: i,
                    success: false,
                    inferenceTime: Date.now() - imageStartTime,
                    error: error.message
                });
            }

            // Progress logging
            if ((i + 1) % 10 === 0) {
                console.log(` Processed ${i + 1}/${images.length} images`);
            }
        }

        const totalTime = Date.now() - startTime;
        const successful = results.filter(r => r.success).length;

        console.log(` Batch processing complete: ${successful}/${images.length} successful in ${totalTime}ms`);

        res.json({ 
            results,
            summary: {
                total: images.length,
                successful: successful,
                failed: images.length - successful,
                totalTime: totalTime,
                averageTime: Math.round(totalTime / images.length)
            }
        });
    } catch (error) {
        console.error(' Batch processing error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Statistics endpoint
app.get('/stats', (req, res) => {
    const stats = {
        modelsLoaded: modelsLoaded,
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        loadingProgress: loadingProgress
    };

    res.json(stats);
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error(' Unhandled error:', error);
    res.status(500).json({ 
        error: 'Internal server error',
        message: error.message 
    });
});

// 404 handler
app.use('*', (req, res) => {
    res.status(404).json({ 
        error: 'Endpoint not found',
        availableEndpoints: ['/health', '/models', '/extract_embedding', '/batch_extract', '/stats']
    });
});

// Server configuration
const PORT = process.env.PORT || 5000;

// Graceful shutdown handling
process.on('SIGINT', () => {
    console.log('\n Received SIGINT, shutting down gracefully...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n Received SIGTERM, shutting down gracefully...');
    process.exit(0);
});

// Start server
console.log(' Starting Face-api.js server...');
console.log(` Port: ${PORT}`);
console.log(` Models path: ${MODEL_URL}`);

loadModels().then((success) => {
    if (success) {
        app.listen(PORT, () => {
            console.log('\n' + '='.repeat(60));
            console.log(' FACE-API.JS SERVER READY');
            console.log('='.repeat(60));
            console.log(` Server URL: http://localhost:${PORT}`);
            console.log(` Health check: http://localhost:${PORT}/health`);
            console.log(` Models status: http://localhost:${PORT}/models`);
            console.log(` Statistics: http://localhost:${PORT}/stats`);
            console.log('='.repeat(60));
        });
    } else {
        console.error(' Failed to load models. Server not started.');
        console.error(' Make sure all model files are in the ./models directory');
        process.exit(1);
    }
}).catch((error) => {
    console.error(' Fatal error during startup:', error);
    process.exit(1);
});
