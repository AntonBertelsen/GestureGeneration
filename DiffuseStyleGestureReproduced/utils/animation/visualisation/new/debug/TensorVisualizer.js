export class TensorVisualizer {
    constructor() {
        this.tensorCanvas = null;
        this.tensorContext = null;
        this.tensors = [];  // Array to hold multiple tensors
        this.currentTensorIndex = 0;
        this.visible = false;
        
        // Initialize the canvas
        this.initTensorCanvas();
    }

    handleTensorMessage(tensorData) {
        this.addTensor(tensorData);
    }
    
    initTensorCanvas() {
        // Create a wrapper div to hold both canvas and controls
        this.canvasContainer = document.createElement('div');
        this.canvasContainer.style.position = 'absolute';
        this.canvasContainer.style.top = '10px';
        this.canvasContainer.style.right = '10px';
        this.canvasContainer.style.display = 'none'; // Hidden by default
        document.body.appendChild(this.canvasContainer);
        
        // Create canvas for tensor visualization
        this.tensorCanvas = document.createElement('canvas');
        this.tensorCanvas.width = 400;
        this.tensorCanvas.height = 300;
        this.tensorCanvas.style.background = 'rgba(0, 0, 0, 0.7)';
        this.tensorCanvas.style.border = '1px solid white';
        this.canvasContainer.appendChild(this.tensorCanvas);
        
        this.tensorContext = this.tensorCanvas.getContext('2d');
        
        // Add navigation controls for multiple tensors
        this.createTensorNavigationControls();
    }

    
    createTensorNavigationControls() {
        // Create controls div and add it to the container, not to the canvas
        const controlsDiv = document.createElement('div');
        controlsDiv.style.position = 'absolute';
        controlsDiv.style.top = '5px';
        controlsDiv.style.left = '5px';
        controlsDiv.style.display = 'flex';
        controlsDiv.style.gap = '5px';
        this.canvasContainer.appendChild(controlsDiv); // Add to container instead of canvas
        
        // Previous tensor button
        const prevButton = document.createElement('button');
        prevButton.innerText = '←';
        prevButton.style.fontSize = '12px';
        prevButton.style.padding = '2px 5px';
        prevButton.onclick = () => this.previousTensor();
        controlsDiv.appendChild(prevButton);
        
        // Next tensor button
        const nextButton = document.createElement('button');
        nextButton.innerText = '→';
        nextButton.style.fontSize = '12px';
        nextButton.style.padding = '2px 5px';
        nextButton.onclick = () => this.nextTensor();
        controlsDiv.appendChild(nextButton);
        
        // Tensor counter/info display
        this.tensorInfoDisplay = document.createElement('span');
        this.tensorInfoDisplay.style.color = 'white';
        this.tensorInfoDisplay.style.fontSize = '12px';
        this.tensorInfoDisplay.style.marginLeft = '5px';
        controlsDiv.appendChild(this.tensorInfoDisplay);
        
        console.log('Tensor navigation controls initialized');

        this.updateTensorInfoDisplay();
    }
    
    updateTensorInfoDisplay() {
        if (!this.tensorInfoDisplay) return;
        
        if (this.tensors.length === 0) {
            this.tensorInfoDisplay.innerText = 'No tensors';
        } else {
            this.tensorInfoDisplay.innerText = `Tensor ${this.currentTensorIndex + 1}/${this.tensors.length}`;
        }
    }
    
    previousTensor() {
        if (this.tensors.length === 0) return;
        
        this.currentTensorIndex = (this.currentTensorIndex - 1 + this.tensors.length) % this.tensors.length;
        this.visualizeCurrentTensor();
        this.updateTensorInfoDisplay();
    }
    
    nextTensor() {
        if (this.tensors.length === 0) return;
        
        this.currentTensorIndex = (this.currentTensorIndex + 1) % this.tensors.length;
        this.visualizeCurrentTensor();
        this.updateTensorInfoDisplay();
    }
    
    addTensor(tensor) {       
        // Check if a tensor with this name already exists
        const existingIndex = this.tensors.findIndex(t => t.name === tensor.name);
        
        if (existingIndex !== -1) {
            // Replace existing tensor with the same name
            this.tensors[existingIndex] = tensor;
            
            // If we're currently viewing the replaced tensor, update the view
            if (existingIndex === this.currentTensorIndex && this.visible) {
                this.visualizeCurrentTensor();
            }
        } else {
            // Add as a new tensor
            this.tensors.push(tensor);
            
            // If this is our first tensor, visualize it
            if (this.tensors.length === 1) {
                this.currentTensorIndex = 0;
                if (this.visible) {
                    this.visualizeCurrentTensor();
                }
            }
        }
        
        this.updateTensorInfoDisplay();
    }
    
    clearTensors() {
        this.tensors = [];
        this.currentTensorIndex = 0;
        this.updateTensorInfoDisplay();
        
        // Clear the canvas
        if (this.tensorContext) {
            this.tensorContext.clearRect(0, 0, this.tensorCanvas.width, this.tensorCanvas.height);
        }
    }
    
    toggleVisibility() {
        return this.setVisible(!this.visible);
    }
    
    setVisible(visible) {
        if (this.visible !== visible) {
            this.visible = visible;
            this.canvasContainer.style.display = visible ? 'block' : 'none';
            
            if (visible && this.tensors.length > 0) {
                this.visualizeCurrentTensor();
            }
        }
        return this.visible;
    }
    
    visualizeCurrentTensor() {
        if (!this.tensorCanvas || !this.tensorContext || this.tensors.length === 0) return;
        const tensor = this.tensors[this.currentTensorIndex];
        this.visualizeTensorData(tensor);
    }
    
    visualizeTensorData(tensorData) {
        if (!this.tensorCanvas || !this.tensorContext || !tensorData) return;

        const { data, name, shape, min_value, max_value } = tensorData;
        const [width, height] = shape;
        
        // Resize the canvas to match the actual tensor dimensions with padding
        const padding = 50;
        this.tensorCanvas.width = width * 2 + padding;
        this.tensorCanvas.height = height * 2 + padding;
        
        // Clear canvas
        this.tensorContext.clearRect(0, 0, this.tensorCanvas.width, this.tensorCanvas.height);
        
        // Draw title and details
        this.tensorContext.fillStyle = 'white';
        this.tensorContext.font = '12px Arial';
        this.tensorContext.fillText(name || `Tensor Visualization (${width}×${height})`, 10, 15);
        this.tensorContext.fillText(`Range: ${min_value.toFixed(4)} to ${max_value.toFixed(4)}`, 10, 30);
        
        // Draw the heatmap
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                // Get normalized value (0 to 1)
                const value = data[y][x];
                const normalizedValue = (value - min_value) / (max_value - min_value || 1);
                
                // Apply viridis colormap
                const color = this.viridisColormap(normalizedValue);
                
                this.tensorContext.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
                this.tensorContext.fillRect(
                    2 * x + padding/2,
                    2 * y + padding/2,
                    2,
                    2
                );
            }
        }
        
        // Draw scale
        const gradientWidth = Math.min(300, this.tensorCanvas.width - 20);
        const gradient = this.tensorContext.createLinearGradient(10, 0, 10 + gradientWidth, 0);
        
        // Create gradient with viridis colormap
        for (let i = 0; i <= 1; i += 0.1) {
            const color = this.viridisColormap(i);
            gradient.addColorStop(i, `rgb(${color.r}, ${color.g}, ${color.b})`);
        }
        
        this.tensorContext.fillStyle = gradient;
        this.tensorContext.fillRect(10, this.tensorCanvas.height - 20, gradientWidth, 10);
        
        // Draw min/max values for the scale
        this.tensorContext.fillStyle = 'white';
        this.tensorContext.font = '10px Arial';
        this.tensorContext.fillText(min_value.toFixed(2), 10, this.tensorCanvas.height - 5);
        this.tensorContext.fillText(max_value.toFixed(2), 10 + gradientWidth - 30, this.tensorCanvas.height - 5);
    }
    
    viridisColormap(value) {
        // Approximation of the viridis colormap
        value = Math.min(1, Math.max(0, value));
        
        let r, g, b;
        
        if (value < 0.25) {
            // Dark purple to blue
            const t = value / 0.25;
            r = Math.round(68 + (46 - 68) * t);
            g = Math.round(1 + (89 - 1) * t);
            b = Math.round(84 + (161 - 84) * t);
        } else if (value < 0.5) {
            // Blue to teal
            const t = (value - 0.25) / 0.25;
            r = Math.round(46 + (31 - 46) * t);
            g = Math.round(89 + (137 - 89) * t);
            b = Math.round(161 + (170 - 161) * t);
        } else if (value < 0.75) {
            // Teal to green
            const t = (value - 0.5) / 0.25;
            r = Math.round(31 + (115 - 31) * t);
            g = Math.round(137 + (171 - 137) * t);
            b = Math.round(170 + (108 - 170) * t);
        } else {
            // Green to yellow
            const t = (value - 0.75) / 0.25;
            r = Math.round(115 + (253 - 115) * t);
            g = Math.round(171 + (231 - 171) * t);
            b = Math.round(108 + (37 - 108) * t);
        }
        
        return { r, g, b };
    }
}