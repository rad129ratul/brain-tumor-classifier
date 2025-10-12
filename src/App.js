import React, { useState, useEffect } from 'react';
import * as ort from 'onnxruntime-web';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

// Configure ONNX Runtime
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

const App = () => {
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'];

  // Load model on component mount
  useEffect(() => {
    loadModel();
  }, []);

  const loadModel = async () => {
    try {
      setLoading(true);
      // Load ONNX model
      const session = await ort.InferenceSession.create('/model/brain_tumor_model.onnx');
      setModel(session);
      setLoading(false);
      console.log('Model loaded successfully');
    } catch (err) {
      console.error('Failed to load model:', err);
      setError('Failed to load model. Please refresh the page.');
      setLoading(false);
    }
  };

  const preprocessImage = async (imageElement) => {
    // Create canvas for image preprocessing
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Resize to 224x224
    canvas.width = 224;
    canvas.height = 224;
    
    // Draw and resize image
    ctx.drawImage(imageElement, 0, 0, 224, 224);
    
    // Get image data
    const imageData = ctx.getImageData(0, 0, 224, 224);
    const data = imageData.data;
    
    // Convert to tensor format (CHW)
    const red = [], green = [], blue = [];
    
    // Normalize using ImageNet statistics
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    for (let i = 0; i < data.length; i += 4) {
      red.push((data[i] / 255.0 - mean[0]) / std[0]);
      green.push((data[i + 1] / 255.0 - mean[1]) / std[1]);
      blue.push((data[i + 2] / 255.0 - mean[2]) / std[2]);
    }
    
    // Combine channels
    const input = Float32Array.from([...red, ...green, ...blue]);
    
    // Create tensor
    const inputTensor = new ort.Tensor('float32', input, [1, 3, 224, 224]);
    return inputTensor;
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setError(null);
      setPrediction(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target.result);
      };
      reader.readAsDataURL(file);
    } else {
      setError('Please select a valid image file');
    }
  };

  const classifyImage = async () => {
    if (!selectedImage || !model) return;
    
    setProcessing(true);
    setError(null);
    
    try {
      // Load image
      const img = new Image();
      img.src = imagePreview;
      
      await new Promise((resolve) => {
        img.onload = resolve;
      });
      
      // Preprocess image
      const inputTensor = await preprocessImage(img);
      
      // Run inference
      const feeds = { input: inputTensor };
      const results = await model.run(feeds);
      
      // Get output
      const output = results.output.data;
      
      // Apply softmax
      const softmax = (arr) => {
        const max = Math.max(...arr);
        const exp = arr.map(x => Math.exp(x - max));
        const sum = exp.reduce((a, b) => a + b);
        return exp.map(x => x / sum);
      };
      
      const probabilities = softmax(Array.from(output));
      
      // Get prediction
      const maxProb = Math.max(...probabilities);
      const predictedClass = probabilities.indexOf(maxProb);
      
      // Calculate uncertainty (entropy)
      const entropy = -probabilities.reduce((sum, p) => {
        return sum + (p > 0 ? p * Math.log(p) : 0);
      }, 0);
      
      // Calculate standard deviation as uncertainty measure
      const mean = probabilities.reduce((a, b) => a + b) / probabilities.length;
      const variance = probabilities.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / probabilities.length;
      const stdDev = Math.sqrt(variance);
      
      setPrediction({
        class: CLASS_NAMES[predictedClass],
        confidence: maxProb * 100,
        probabilities: probabilities.map((p, i) => ({
          class: CLASS_NAMES[i],
          probability: p * 100
        })),
        uncertainty: {
          entropy: entropy.toFixed(4),
          stdDev: stdDev.toFixed(4)
        }
      });
      
    } catch (err) {
      console.error('Classification error:', err);
      setError('Failed to classify image. Please try again.');
    } finally {
      setProcessing(false);
    }
  };

  const resetApp = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setPrediction(null);
    setError(null);
    document.getElementById('imageInput').value = '';
  };

  return (
    <div className="min-vh-100 bg-light">
      {/* Header */}
      <nav className="navbar navbar-dark bg-primary shadow-sm">
        <div className="container">
          <span className="navbar-brand mb-0 h1">
            🧠 Brain Tumor Classifier
          </span>
          <span className="text-white">
            AI-Powered MRI Analysis
          </span>
        </div>
      </nav>

      {/* Main Content */}
      <div className="container py-5">
        <div className="row justify-content-center">
          <div className="col-lg-8">
            {/* Info Card */}
            <div className="card shadow-sm mb-4">
              <div className="card-body">
                <h5 className="card-title">About This Tool</h5>
                <p className="card-text">
                  Upload a brain MRI image to classify potential tumors using our advanced AI model.
                  The system can identify: <strong>Glioma</strong>, <strong>Meningioma</strong>, 
                  <strong>Pituitary tumor</strong>, or <strong>No tumor</strong>.
                </p>
                <div className="alert alert-info mb-0">
                  <small>
                    ⚠️ This tool is for educational purposes only. Always consult healthcare professionals for medical diagnosis.
                  </small>
                </div>
              </div>
            </div>

            {/* Upload Card */}
            <div className="card shadow">
              <div className="card-body">
                <h5 className="card-title mb-4">Upload MRI Image</h5>
                
                {loading ? (
                  <div className="text-center py-5">
                    <div className="spinner-border text-primary" role="status">
                      <span className="visually-hidden">Loading model...</span>
                    </div>
                    <p className="mt-3">Loading AI model...</p>
                  </div>
                ) : (
                  <>
                    {/* File Input */}
                    <div className="mb-4">
                      <input
                        type="file"
                        className="form-control"
                        id="imageInput"
                        accept="image/*"
                        onChange={handleImageUpload}
                        disabled={processing}
                      />
                    </div>

                    {/* Image Preview */}
                    {imagePreview && (
                      <div className="text-center mb-4">
                        <img 
                          src={imagePreview} 
                          alt="MRI Preview" 
                          className="img-fluid rounded shadow-sm"
                          style={{ maxHeight: '300px' }}
                        />
                      </div>
                    )}

                    {/* Action Buttons */}
                    {selectedImage && (
                      <div className="d-grid gap-2 mb-4">
                        <button 
                          className="btn btn-primary"
                          onClick={classifyImage}
                          disabled={processing}
                        >
                          {processing ? (
                            <>
                              <span className="spinner-border spinner-border-sm me-2" />
                              Processing...
                            </>
                          ) : (
                            'Analyze Image'
                          )}
                        </button>
                        <button 
                          className="btn btn-outline-secondary"
                          onClick={resetApp}
                          disabled={processing}
                        >
                          Clear & Start Over
                        </button>
                      </div>
                    )}

                    {/* Error Display */}
                    {error && (
                      <div className="alert alert-danger" role="alert">
                        {error}
                      </div>
                    )}

                    {/* Prediction Results */}
                    {prediction && (
                      <div className="mt-4">
                        <h5 className="mb-3">Analysis Results</h5>
                        
                        {/* Primary Prediction */}
                        <div className="alert alert-success">
                          <h6>Predicted Class: <strong>{prediction.class}</strong></h6>
                          <div className="progress mt-2" style={{ height: '25px' }}>
                            <div 
                              className="progress-bar bg-success"
                              style={{ width: `${prediction.confidence}%` }}
                            >
                              {prediction.confidence.toFixed(1)}% Confidence
                            </div>
                          </div>
                        </div>

                        {/* All Probabilities */}
                        <div className="card mb-3">
                          <div className="card-body">
                            <h6 className="card-title">Class Probabilities</h6>
                            {prediction.probabilities.map((item, idx) => (
                              <div key={idx} className="mb-2">
                                <div className="d-flex justify-content-between mb-1">
                                  <small>{item.class}</small>
                                  <small>{item.probability.toFixed(2)}%</small>
                                </div>
                                <div className="progress" style={{ height: '10px' }}>
                                  <div 
                                    className={`progress-bar ${
                                      item.class === prediction.class ? 'bg-primary' : 'bg-secondary'
                                    }`}
                                    style={{ width: `${item.probability}%` }}
                                  />
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Uncertainty Metrics */}
                        <div className="card">
                          <div className="card-body">
                            <h6 className="card-title">Uncertainty Metrics</h6>
                            <div className="row">
                              <div className="col-6">
                                <small className="text-muted">Entropy</small>
                                <p className="mb-0"><strong>{prediction.uncertainty.entropy}</strong></p>
                              </div>
                              <div className="col-6">
                                <small className="text-muted">Std Deviation</small>
                                <p className="mb-0"><strong>{prediction.uncertainty.stdDev}</strong></p>
                              </div>
                            </div>
                            <small className="text-muted mt-2 d-block">
                              Lower values indicate higher confidence in the prediction
                            </small>
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-dark text-white text-center py-3 mt-5">
        <small>
          Brain Tumor Classifier v1.0 | Model Accuracy: 98.38% | Built with React & ONNX Runtime
        </small>
      </footer>
    </div>
  );
};

export default App;