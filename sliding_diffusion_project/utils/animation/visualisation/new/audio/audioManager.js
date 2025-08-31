export class AudioManager {
    constructor() {
        this.audioContext = null;
        this.currentSource = null;
        this.isMuted = false;
        this.initAudioContext();
    }

    initAudioContext() {
        // Create audio context - it will be suspended until first user interaction
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            console.log("Audio context initialized");
        } catch (error) {
            console.error("Failed to initialize audio context:", error);
        }
    }

    handleAudioMessage(data) {
        if (this.isMuted) {
            console.log("Audio is muted, skipping playback");
            return;
        }

        try {
            const audioBase64 = data.audio;
            if (!audioBase64) {
                console.warn("No audio data in message");
                return;
            }
            
            this.playAudio(audioBase64);
        } catch (error) {
            console.error("Error handling audio message:", error);
        }
    }

    playAudio(audioBase64) {
        if (!this.audioContext) {
            console.warn("Audio context not initialized");
            return;
        }

        // Resume audio context if it's suspended
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }

        try {
            // Decode base64 audio data
            const binaryString = atob(audioBase64);
            const byteArray = new Uint8Array(binaryString.length);
            
            for (let i = 0; i < binaryString.length; i++) {
                byteArray[i] = binaryString.charCodeAt(i);
            }
            
            // Convert to AudioBuffer
            this.audioContext.decodeAudioData(byteArray.buffer)
                .then(audioBuffer => {
                    // Create audio source
                    const source = this.audioContext.createBufferSource();
                    source.buffer = audioBuffer;
                    
                    // Create a more gradual fade-in to avoid pops
                    const gainNode = this.audioContext.createGain();
                    const fadeInTime = 0.1; // Longer fade-in (150ms)
                    gainNode.gain.setValueAtTime(0, this.audioContext.currentTime);
                    gainNode.gain.linearRampToValueAtTime(1, this.audioContext.currentTime + fadeInTime);
                    
                    source.connect(gainNode);
                    gainNode.connect(this.audioContext.destination);
                    
                    // If there's a current source playing, create a more gradual crossfade
                    if (this.currentSource) {
                        const oldSource = this.currentSource;
                        const oldGain = oldSource.gainNode;
                        
                        // Longer fade-out (150ms) with an exponential curve for more natural fading
                        oldGain.gain.setValueAtTime(oldGain.gain.value, this.audioContext.currentTime);
                        oldGain.gain.exponentialRampToValueAtTime(0.001, this.audioContext.currentTime + fadeInTime);
                        
                        setTimeout(() => {
                            try {
                                oldSource.stop();
                            } catch (e) {
                                // Suppress errors if already stopped
                            }
                        }, fadeInTime * 1000 + 50); // Convert to ms and add small buffer
                    }
                    
                    // Save reference with its gain node
                    this.currentSource = source;
                    this.currentSource.gainNode = gainNode;
                    
                    // Play immediately
                    source.start(0);
                    
                    // Clean up when done
                    source.onended = () => {
                        if (this.currentSource === source) {
                            this.currentSource = null;
                        }
                    };
                })
                .catch(err => {
                    console.error("Error decoding audio data:", err);
                });
        } catch (error) {
            console.error("Error processing audio data:", error);
        }
    }

    stopCurrentAudio() {
        if (this.currentSource) {
            try {
                // Fade out before stopping
                const gainNode = this.currentSource.gainNode || 
                                 this.currentSource.connect(this.audioContext.createGain());
                gainNode.gain.linearRampToValueAtTime(0, this.audioContext.currentTime + 0.01);
                
                const source = this.currentSource;
                this.currentSource = null;
                
                // Stop after fade completes
                setTimeout(() => {
                    source.stop();
                }, 15);
            } catch (error) {
                console.warn("Error stopping current audio:", error);
            }
        }
    }

    toggleMute() {
        this.isMuted = !this.isMuted;
        
        if (this.isMuted) {
            this.stopCurrentAudio();
        }
        
        return this.isMuted;
    }

    // Call this method to add a mute button to the UI
    addMuteButtonToUI(controlsContainer) {
        const muteButton = document.createElement('button');
        muteButton.innerText = 'Mute Audio';
        muteButton.style.marginLeft = '5px';
        
        muteButton.onclick = () => {
            const isMuted = this.toggleMute();
            muteButton.innerText = isMuted ? 'Unmute Audio' : 'Mute Audio';
        };
        
        controlsContainer.appendChild(muteButton);
    }
}