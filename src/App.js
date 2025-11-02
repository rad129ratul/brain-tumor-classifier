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

  // Updated class names matching the fine-tuned model
  // Maps to: ['glioma', 'meningioma', 'notumor', 'pituitary']
  const CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'];
  
  // Model performance metrics from fine-tuning
  const MODEL_METRICS = {
    accuracy: 98.10,
    originalAccuracy: 30.48,
    improvement: 67.62,
    dataset: 'External Validation (3K images)',
    fineTuned: true
  };

  // Load model on component mount
  useEffect(() => {
    loadModel();
  }, []);

  const loadModel = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Load fine-tuned ONNX model
      const session = await ort.InferenceSession.create('/model/brain_tumor_model.onnx', {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      });
      
      setModel(session);
      setLoading(false);
      console.log('Fine-tuned model loaded successfully');
      console.log('Input shape:', session.inputNames);
      console.log('Output shape:', session.outputNames);
    } catch (err) {
      console.error('Failed to load model:', err);
      setError('Failed to load the AI model. Please refresh the page or check if the model file exists.');
      setLoading(false);
    }
  };

  const preprocessImage = async (imageElement) => {
    // Create canvas for image preprocessing
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Resize to 224x224 (as per model training)
    canvas.width = 224;
    canvas.height = 224;
    
    // Draw and resize image
    ctx.drawImage(imageElement, 0, 0, 224, 224);
    
    // Get image data
    const imageData = ctx.getImageData(0, 0, 224, 224);
    const data = imageData.data;
    
    // Convert to tensor format (NCHW: batch, channels, height, width)
    const red = [], green = [], blue = [];
    
    // Normalize using ImageNet statistics (same as training)
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    for (let i = 0; i < data.length; i += 4) {
      // Normalize each channel
      red.push((data[i] / 255.0 - mean[0]) / std[0]);
      green.push((data[i + 1] / 255.0 - mean[1]) / std[1]);
      blue.push((data[i + 2] / 255.0 - mean[2]) / std[2]);
    }
    
    // Combine channels in CHW format
    const input = Float32Array.from([...red, ...green, ...blue]);
    
    // Create tensor with shape [1, 3, 224, 224]
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
      setError('Please select a valid image file (JPEG, PNG)');
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
      
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });
      
      // Preprocess image
      const inputTensor = await preprocessImage(img);
      
      // Run inference with the fine-tuned model
      const feeds = { input: inputTensor };
      const results = await model.run(feeds);
      
      // Get output (logits)
      const output = results.output.data;
      
      // Apply softmax to get probabilities
      const softmax = (arr) => {
        const max = Math.max(...arr);
        const exp = arr.map(x => Math.exp(x - max));
        const sum = exp.reduce((a, b) => a + b, 0);
        return exp.map(x => x / sum);
      };
      
      const probabilities = softmax(Array.from(output));
      
      // Get prediction
      const maxProb = Math.max(...probabilities);
      const predictedClass = probabilities.indexOf(maxProb);
      
      // Calculate uncertainty metrics
      // Entropy: measure of prediction uncertainty
      const entropy = -probabilities.reduce((sum, p) => {
        return sum + (p > 0 ? p * Math.log(p) : 0);
      }, 0);
      
      // Standard deviation: measure of probability spread
      const mean = probabilities.reduce((a, b) => a + b, 0) / probabilities.length;
      const variance = probabilities.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / probabilities.length;
      const stdDev = Math.sqrt(variance);
      
      // Determine confidence level
      const getConfidenceLevel = (confidence) => {
        if (confidence >= 95) return { level: 'Very High', color: 'success' };
        if (confidence >= 85) return { level: 'High', color: 'info' };
        if (confidence >= 70) return { level: 'Moderate', color: 'warning' };
        return { level: 'Low', color: 'danger' };
      };
      
      const confidenceInfo = getConfidenceLevel(maxProb * 100);
      
      setPrediction({
        class: CLASS_NAMES[predictedClass],
        classIndex: predictedClass,
        confidence: maxProb * 100,
        confidenceLevel: confidenceInfo.level,
        confidenceColor: confidenceInfo.color,
        probabilities: probabilities.map((p, i) => ({
          class: CLASS_NAMES[i],
          probability: p * 100,
          isTop: i === predictedClass
        })),
        uncertainty: {
          entropy: entropy,
          entropyFormatted: entropy.toFixed(4),
          stdDev: stdDev,
          stdDevFormatted: stdDev.toFixed(4),
          uncertaintyLevel: entropy < 0.5 ? 'Low' : entropy < 1.0 ? 'Moderate' : 'High'
        }
      });
      
      console.log('Prediction:', CLASS_NAMES[predictedClass]);
      console.log('Confidence:', (maxProb * 100).toFixed(2) + '%');
      console.log('Probabilities:', probabilities.map(p => (p * 100).toFixed(2) + '%'));
      
    } catch (err) {
      console.error('Classification error:', err);
      setError('Failed to classify image. Please ensure the image is a valid brain MRI scan and try again.');
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
            🧠 Brain Tumor Classifier - Fine-tuned Model
          </span>
          <span className="text-white small">
            Bayesian MobileNetV2 | Domain Adapted
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
                <h5 className="card-title">
                  <span className="badge bg-success me-2">Fine-tuned</span>
                  About This Tool
                </h5>
                <p className="card-text">
                  Upload a brain MRI image to classify potential tumors using our advanced fine-tuned AI model.
                  The system can identify: <strong>Glioma</strong>, <strong>Meningioma</strong>, 
                  <strong>Pituitary tumor</strong>, or <strong>No tumor</strong>.
                </p>
                
                {/* Model Performance Metrics */}
                <div className="row g-2 mb-3">
                  <div className="col-md-3">
                    <div className="card bg-success text-white">
                      <div className="card-body text-center p-2">
                        <small className="d-block">Accuracy</small>
                        <strong>{MODEL_METRICS.accuracy}%</strong>
                      </div>
                    </div>
                  </div>
                  <div className="col-md-3">
                    <div className="card bg-info text-white">
                      <div className="card-body text-center p-2">
                        <small className="d-block">Improvement</small>
                        <strong>+{MODEL_METRICS.improvement}%</strong>
                      </div>
                    </div>
                  </div>
                  <div className="col-md-6">
                    <div className="card bg-secondary text-white">
                      <div className="card-body text-center p-2">
                        <small className="d-block">Validation Dataset</small>
                        <strong>{MODEL_METRICS.dataset}</strong>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="alert alert-info mb-0">
                  <small>
                    ⚠️ <strong>Important:</strong> This tool is for educational and research purposes only. 
                    Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.
                  </small>
                </div>
              </div>
            </div>

            {/* Upload Card */}
            <div className="card shadow">
              <div className="card-body">
                <h5 className="card-title mb-4">
                  Upload Brain MRI Image
                  <span className="badge bg-primary ms-2 small">Step 1</span>
                </h5>
                
                {loading ? (
                  <div className="text-center py-5">
                    <div className="spinner-border text-primary" role="status" style={{ width: '3rem', height: '3rem' }}>
                      <span className="visually-hidden">Loading model...</span>
                    </div>
                    <p className="mt-3 text-muted">Loading fine-tuned AI model...</p>
                    <small className="text-muted">This may take a few moments</small>
                  </div>
                ) : (
                  <>
                    {/* File Input */}
                    <div className="mb-4">
                      <label htmlFor="imageInput" className="form-label">
                        Select MRI Image
                      </label>
                      <input
                        type="file"
                        className="form-control form-control-lg"
                        id="imageInput"
                        accept="image/*"
                        onChange={handleImageUpload}
                        disabled={processing}
                      />
                      <small className="form-text text-muted">
                        Supported formats: JPEG, PNG | Recommended: Brain MRI scans
                      </small>
                    </div>

                    {/* Image Preview */}
                    {imagePreview && (
                      <div className="mb-4">
                        <h6 className="mb-3">
                          Image Preview
                          <span className="badge bg-primary ms-2 small">Step 2</span>
                        </h6>
                        <div className="text-center bg-dark rounded p-3">
                          <img 
                            src={imagePreview} 
                            alt="MRI Preview" 
                            className="img-fluid rounded shadow"
                            style={{ maxHeight: '400px', objectFit: 'contain' }}
                          />
                        </div>
                      </div>
                    )}

                    {/* Action Buttons */}
                    {selectedImage && (
                      <div className="mb-4">
                        <h6 className="mb-3">
                          Actions
                          <span className="badge bg-primary ms-2 small">Step 3</span>
                        </h6>
                        <div className="d-grid gap-2">
                          <button 
                            className="btn btn-primary btn-lg"
                            onClick={classifyImage}
                            disabled={processing}
                          >
                            {processing ? (
                              <>
                                <span className="spinner-border spinner-border-sm me-2" />
                                Analyzing with Bayesian Model...
                              </>
                            ) : (
                              <>
                                🔍 Analyze Image with AI
                              </>
                            )}
                          </button>
                          <button 
                            className="btn btn-outline-secondary"
                            onClick={resetApp}
                            disabled={processing}
                          >
                            🔄 Clear & Upload New Image
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Error Display */}
                    {error && (
                      <div className="alert alert-danger d-flex align-items-center" role="alert">
                        <svg className="bi flex-shrink-0 me-2" width="24" height="24" role="img" aria-label="Danger:">
                          <use xlinkHref="#exclamation-triangle-fill"/>
                        </svg>
                        <div>{error}</div>
                      </div>
                    )}

                    {/* Prediction Results */}
                    {prediction && (
                      <div className="mt-4">
                        <h5 className="mb-3">
                          Analysis Results
                          <span className="badge bg-success ms-2">Complete</span>
                        </h5>
                        
                        {/* Primary Prediction */}
                        <div className={`alert alert-${prediction.confidenceColor} shadow-sm`}>
                          <div className="d-flex justify-content-between align-items-center mb-2">
                            <h6 className="mb-0">
                              <strong>Predicted Diagnosis:</strong> {prediction.class}
                            </h6>
                            <span className={`badge bg-${prediction.confidenceColor}`}>
                              {prediction.confidenceLevel} Confidence
                            </span>
                          </div>
                          <div className="progress" style={{ height: '30px' }}>
                            <div 
                              className={`progress-bar progress-bar-striped progress-bar-animated bg-${prediction.confidenceColor}`}
                              style={{ width: `${prediction.confidence}%` }}
                            >
                              <strong>{prediction.confidence.toFixed(2)}%</strong>
                            </div>
                          </div>
                          <small className="text-muted mt-2 d-block">
                            Model confidence in this prediction
                          </small>
                        </div>

                        {/* All Probabilities */}
                        <div className="card mb-3 shadow-sm">
                          <div className="card-body">
                            <h6 className="card-title">
                              <svg width="20" height="20" fill="currentColor" className="me-2" viewBox="0 0 16 16">
                                <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14zm0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16z"/>
                                <path d="M7.002 11a1 1 0 1 1 2 0 1 1 0 0 1-2 0zM7.1 4.995a.905.905 0 1 1 1.8 0l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 4.995z"/>
                              </svg>
                              Class Probabilities (All Categories)
                            </h6>
                            {prediction.probabilities.map((item, idx) => (
                              <div key={idx} className="mb-3">
                                <div className="d-flex justify-content-between mb-1">
                                  <span className={item.isTop ? 'fw-bold' : ''}>
                                    {item.class}
                                    {item.isTop && <span className="badge bg-primary ms-2 small">Predicted</span>}
                                  </span>
                                  <span className={item.isTop ? 'fw-bold text-primary' : 'text-muted'}>
                                    {item.probability.toFixed(2)}%
                                  </span>
                                </div>
                                <div className="progress" style={{ height: '15px' }}>
                                  <div 
                                    className={`progress-bar ${
                                      item.isTop ? 'bg-primary' : 'bg-secondary'
                                    }`}
                                    style={{ width: `${item.probability}%` }}
                                  />
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Uncertainty Metrics */}
                        <div className="card shadow-sm">
                          <div className="card-body">
                            <h6 className="card-title">
                              <svg width="20" height="20" fill="currentColor" className="me-2" viewBox="0 0 16 16">
                                <path d="M8 16A8 8 0 1 0 8 0a8 8 0 0 0 0 16zM7 6.5C7 7.328 6.552 8 6 8s-1-.672-1-1.5S5.448 5 6 5s1 .672 1 1.5zM4.285 9.567a.5.5 0 0 1 .683.183A3.498 3.498 0 0 0 8 11.5a3.498 3.498 0 0 0 3.032-1.75.5.5 0 1 1 .866.5A4.498 4.498 0 0 1 8 12.5a4.498 4.498 0 0 1-3.898-2.25.5.5 0 0 1 .183-.683zM10 8c-.552 0-1-.672-1-1.5S9.448 5 10 5s1 .672 1 1.5S10.552 8 10 8z"/>
                              </svg>
                              Bayesian Uncertainty Metrics
                              <span className={`badge ms-2 ${
                                prediction.uncertainty.uncertaintyLevel === 'Low' ? 'bg-success' :
                                prediction.uncertainty.uncertaintyLevel === 'Moderate' ? 'bg-warning' : 'bg-danger'
                              }`}>
                                {prediction.uncertainty.uncertaintyLevel} Uncertainty
                              </span>
                            </h6>
                            <div className="row g-3 mt-2">
                              <div className="col-md-6">
                                <div className="card bg-light">
                                  <div className="card-body">
                                    <small className="text-muted d-block">Predictive Entropy</small>
                                    <h5 className="mb-0">{prediction.uncertainty.entropyFormatted}</h5>
                                    <small className="text-muted">
                                      Measures prediction uncertainty
                                    </small>
                                  </div>
                                </div>
                              </div>
                              <div className="col-md-6">
                                <div className="card bg-light">
                                  <div className="card-body">
                                    <small className="text-muted d-block">Standard Deviation</small>
                                    <h5 className="mb-0">{prediction.uncertainty.stdDevFormatted}</h5>
                                    <small className="text-muted">
                                      Probability distribution spread
                                    </small>
                                  </div>
                                </div>
                              </div>
                            </div>
                            <div className="alert alert-light mt-3 mb-0">
                              <small>
                                <strong>Interpretation:</strong> Lower uncertainty values indicate higher model confidence. 
                                The Bayesian approach provides probabilistic predictions with uncertainty estimates, 
                                helping assess prediction reliability.
                              </small>
                            </div>
                          </div>
                        </div>

                        {/* Clinical Note */}
                        <div className="alert alert-warning mt-3 mb-0">
                          <strong>⚕️ Clinical Note:</strong>
                          <ul className="mb-0 mt-2 small">
                            <li>This AI prediction should be used as a decision support tool only</li>
                            <li>Final diagnosis must be made by qualified radiologists and physicians</li>
                            <li>Consider clinical history, symptoms, and additional diagnostic tests</li>
                            <li>High uncertainty scores warrant additional clinical review</li>
                          </ul>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>

            {/* Technical Details Card */}
            <div className="card shadow-sm mt-4">
              <div className="card-body">
                <h6 className="card-title">Technical Details</h6>
                <div className="row small">
                  <div className="col-md-6">
                    <strong>Model Architecture:</strong>
                    <p className="mb-1">Bayesian MobileNetV2 with Attention</p>
                  </div>
                  <div className="col-md-6">
                    <strong>Training Dataset:</strong>
                    <p className="mb-1">7,023 brain MRI images (4 classes)</p>
                  </div>
                  <div className="col-md-6">
                    <strong>Fine-tuning Dataset:</strong>
                    <p className="mb-1">3,264 external validation images</p>
                  </div>
                  <div className="col-md-6">
                    <strong>Input Preprocessing:</strong>
                    <p className="mb-1">224×224, ImageNet normalization</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-dark text-white py-4 mt-5">
        <div className="container">
          <div className="row">
            <div className="col-md-8">
              <h6>Brain Tumor Classifier - Fine-tuned Model</h6>
              <p className="small mb-0">
                Bayesian MobileNetV2 | Test Accuracy: {MODEL_METRICS.accuracy}% | 
                Domain Adapted | Built with React & ONNX Runtime
              </p>
            </div>
            <div className="col-md-4 text-md-end">
              <small className="text-muted">
                Model Version: 1.1 (Fine-tuned)<br/>
                Last Updated: 2025
              </small>
            </div>
          </div>
        </div>
      </footer>

      {/* SVG Icons */}
      <svg xmlns="http://www.w3.org/2000/svg" style={{ display: 'none' }}>
        <symbol id="exclamation-triangle-fill" fill="currentColor" viewBox="0 0 16 16">
          <path d="M8.982 1.566a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767L8.982 1.566zM8 5c.535 0 .954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 0 0 1 8 5zm.002 6a1 1 0 1 1 0 2 1 1 0 0 1 0-2z"/>
        </symbol>
      </svg>
    </div>
  );
};

export default App;