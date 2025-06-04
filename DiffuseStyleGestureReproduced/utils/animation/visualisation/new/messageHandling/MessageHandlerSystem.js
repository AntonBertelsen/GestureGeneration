// Message handling system
export class MessageHandlerSystem {
    constructor() {
        this.handlers = {};
        this.processors = {};
    }
    
    registerHandler(messageType, handler) {
        this.handlers[messageType] = handler;
    }
    
    registerProcessor(messageType, processor) {
        this.processors[messageType] = processor;
    }
    
    processMessage(message) {
        try {
            const data = JSON.parse(message);
            
            // Process each message component with its registered handler
            Object.keys(data).forEach(key => {
                if (this.handlers[key]) {
                    this.handlers[key](data[key]);
                }
                
                if (this.processors[key]) {
                    // Process with transformations if needed
                    const processed = this.processors[key](data[key]);
                    if (processed && this.handlers[key]) {
                        this.handlers[key](processed);
                    }
                }
            });
            
        } catch (error) {
            console.error('Error processing message:', error);
        }
    }
}