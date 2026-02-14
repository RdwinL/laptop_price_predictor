// Update brand-specific options when company changes
document.getElementById('company').addEventListener('change', function() {
    const company = this.value;
    
    fetch('/get_brand_options', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ company: company })
    })
    .then(response => response.json())
    .then(data => {
        // Update type options
        const typeSelect = document.getElementById('type');
        typeSelect.innerHTML = '';
        data.types.forEach(type => {
            const option = document.createElement('option');
            option.value = type;
            option.textContent = type;
            typeSelect.appendChild(option);
        });
        
        // Update OS options
        const osSelect = document.getElementById('os');
        osSelect.innerHTML = '';
        data.os.forEach(os => {
            const option = document.createElement('option');
            option.value = os;
            option.textContent = os;
            osSelect.appendChild(option);
        });
        
        // Update CPU options
        const cpuSelect = document.getElementById('cpu_company');
        cpuSelect.innerHTML = '';
        data.cpu.forEach(cpu => {
            const option = document.createElement('option');
            option.value = cpu;
            option.textContent = cpu;
            cpuSelect.appendChild(option);
        });
        
        // Update GPU options
        const gpuSelect = document.getElementById('gpu_company');
        gpuSelect.innerHTML = '';
        data.gpu.forEach(gpu => {
            const option = document.createElement('option');
            option.value = gpu;
            option.textContent = gpu;
            gpuSelect.appendChild(option);
        });
    });
});

// Handle prediction form submission
document.getElementById('predictionForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const data = {};
    formData.forEach((value, key) => {
        data[key] = value;
    });
    
    // Show loading state
    const submitBtn = this.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Predicting...';
    
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
        
        if (result.success) {
            // Store prediction data for PDF download
            window.currentPrediction = {
                prediction: result.prediction,
                prediction_usd: result.prediction_usd,
                specs: {
                    'Brand': data.company,
                    'Type': data.type,
                    'Screen Size': data.inches + '"',
                    'RAM': data.ram + ' GB',
                    'Storage': data.storage + ' GB',
                    'Operating System': data.os,
                    'CPU Brand': data.cpu_company,
                    'CPU Frequency': data.cpu_freq + ' GHz',
                    'GPU Brand': data.gpu_company,
                    'Weight': data.weight + ' kg',
                    'Touchscreen': data.touchscreen,
                    'IPS Panel': data.ips_panel,
                    'Retina Display': data.retina_display
                }
            };
            
            // Display prediction
            document.getElementById('predictedPrice').textContent = result.prediction.toLocaleString() + ' Tsh';
            document.getElementById('predictedPriceUSD').textContent = '≈ $' + result.prediction_usd + ' USD';
            document.getElementById('predictionResult').style.display = 'block';
            
            // Display recommendations
            if (result.recommendations && result.recommendations.length > 0) {
                let recsHTML = '<div class="section-header"><h4>Similar Products from ' + data.company + '</h4></div>';
                
                result.recommendations.forEach((rec, index) => {
                    const diff = rec.price - result.prediction;
                    const diffPercent = ((diff / result.prediction) * 100).toFixed(1);
                    const diffColor = diff > 0 ? '#e74c3c' : '#27ae60';
                    const diffSymbol = diff > 0 ? '+' : '';
                    
                    recsHTML += `
                        <div class="product-card">
                            <div class="d-flex justify-content-between align-items-center">
                                <h4 class="mb-0" style="color: #2980b9;">${index + 1}. ${rec.name}</h4>
                                <span class="badge" style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); 
                                                          font-size: 0.95rem; padding: 0.5rem 1rem;">
                                    ${rec.price.toLocaleString()} Tsh
                                </span>
                            </div>
                            
                            <div class="row mt-3 pt-3" style="border-top: 2px solid #ebf5fb;">
                                <div class="col-md-3 text-center p-2" style="background: #f8f9fa; border-radius: 6px; margin: 0 0.25rem;">
                                    <small class="text-muted text-uppercase">Type</small>
                                    <div class="fw-bold" style="color: #2c3e50;">${rec.type}</div>
                                </div>
                                <div class="col-md-2 text-center p-2" style="background: #f8f9fa; border-radius: 6px; margin: 0 0.25rem;">
                                    <small class="text-muted text-uppercase">RAM</small>
                                    <div class="fw-bold" style="color: #2c3e50;">${rec.ram} GB</div>
                                </div>
                                <div class="col-md-3 text-center p-2" style="background: #f8f9fa; border-radius: 6px; margin: 0 0.25rem;">
                                    <small class="text-muted text-uppercase">Storage</small>
                                    <div class="fw-bold" style="color: #2c3e50;">${rec.storage} GB</div>
                                </div>
                                <div class="col-md-2 text-center p-2" style="background: #f8f9fa; border-radius: 6px; margin: 0 0.25rem;">
                                    <small class="text-muted text-uppercase">Screen</small>
                                    <div class="fw-bold" style="color: #2c3e50;">${rec.screen}"</div>
                                </div>
                            </div>
                            
                            <div class="mt-3 p-3" style="background: #ebf5fb; border-radius: 6px; 
                                                        display: flex; justify-content: space-between; align-items: center;">
                                <span style="color: #2c3e50; font-size: 0.9rem;">
                                    <strong>Price vs Prediction:</strong>
                                </span>
                                <span style="color: ${diffColor}; font-weight: 600; font-size: 1rem;">
                                    ${diffSymbol}${Math.abs(diff).toLocaleString()} Tsh (${diffSymbol}${diffPercent}%)
                                </span>
                            </div>
                        </div>
                    `;
                });
                
                document.getElementById('recommendations').innerHTML = recsHTML;
            }
            
            // Scroll to result
            document.getElementById('predictionResult').scrollIntoView({ behavior: 'smooth' });
        } else {
            alert('Error: ' + result.error);
        }
    })
    .catch(error => {
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
        alert('Error: ' + error);
    });
});

// Handle budget search form submission
document.getElementById('budgetForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const data = {};
    formData.forEach((value, key) => {
        if (key.includes('price') || key.includes('ram')) {
            data[key] = parseInt(value);
        } else {
            data[key] = value;
        }
    });
    
    const submitBtn = this.querySelector('button[type="submit"]');
    const originalText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Searching...';
    
    fetch('/budget_search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
        
        if (result.success) {
            let resultsHTML = `
                <div class="alert alert-info">
                    <h4 class="mb-2">Search Results: ${result.total} laptops found</h4>
                    <p class="mb-0">Budget Range: ${data.min_price.toLocaleString()} - ${data.max_price.toLocaleString()} Tsh</p>
                </div>
            `;
            
            if (result.laptops.length > 0) {
                // Statistics
                resultsHTML += '<div class="row mb-4">';
                resultsHTML += `
                    <div class="col-md-3">
                        <div class="metric-card">
                            <p class="stat-label">Average Price</p>
                            <p class="stat-value" style="font-size: 1.5rem;">${(result.stats.avg_price / 1000).toFixed(0)}K</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <p class="stat-label">Avg RAM</p>
                            <p class="stat-value" style="font-size: 1.5rem;">${result.stats.avg_ram} GB</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <p class="stat-label">Avg Storage</p>
                            <p class="stat-value" style="font-size: 1.5rem;">${result.stats.avg_storage} GB</p>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="metric-card">
                            <p class="stat-label">Avg Screen</p>
                            <p class="stat-value" style="font-size: 1.5rem;">${result.stats.avg_screen}"</p>
                        </div>
                    </div>
                `;
                resultsHTML += '</div>';
                
                // Laptops list
                resultsHTML += '<div class="section-header"><h4>Available Laptops</h4></div>';
                
                result.laptops.forEach((laptop, index) => {
                    resultsHTML += `
                        <div class="product-card">
                            <h4 style="margin-top: 0; color: #2980b9;">${laptop.name}</h4>
                            <div class="row mt-3">
                                <div class="col-md-3">
                                    <p class="mb-1"><strong>Price:</strong> ${laptop.price.toLocaleString()} Tsh</p>
                                    <p class="mb-0 text-muted">≈ $${laptop.price_usd} USD</p>
                                </div>
                                <div class="col-md-3">
                                    <p class="mb-1"><strong>Type:</strong> ${laptop.type}</p>
                                    <p class="mb-0"><strong>OS:</strong> ${laptop.os}</p>
                                </div>
                                <div class="col-md-3">
                                    <p class="mb-1"><strong>RAM:</strong> ${laptop.ram} GB</p>
                                    <p class="mb-0"><strong>Storage:</strong> ${laptop.storage} GB</p>
                                </div>
                                <div class="col-md-3">
                                    <p class="mb-1"><strong>Screen:</strong> ${laptop.screen}"</p>
                                    <p class="mb-0"><strong>CPU:</strong> ${laptop.cpu}</p>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                if (result.total > 15) {
                    resultsHTML += `<div class="alert alert-info mt-3">Showing top 15 of ${result.total} results.</div>`;
                }
            } else {
                resultsHTML += '<div class="alert alert-warning">No laptops found. Try adjusting your criteria.</div>';
            }
            
            document.getElementById('budgetResults').innerHTML = resultsHTML;
            document.getElementById('budgetResults').style.display = 'block';
        } else {
            alert('Error: ' + result.error);
        }
    })
    .catch(error => {
        submitBtn.disabled = false;
        submitBtn.textContent = originalText;
        alert('Error: ' + error);
    });
});

// Download PDF report
function downloadReport() {
    if (!window.currentPrediction) {
        alert('No prediction data available');
        return;
    }
    
    fetch('/download_report', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(window.currentPrediction)
    })
    .then(response => response.blob())
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `laptop_price_prediction_${new Date().getTime()}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
    })
    .catch(error => alert('Error downloading report: ' + error));
}

// Reset form
function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('predictionResult').style.display = 'none';
    window.currentPrediction = null;
}
