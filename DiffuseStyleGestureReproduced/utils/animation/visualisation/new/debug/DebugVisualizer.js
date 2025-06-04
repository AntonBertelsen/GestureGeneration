import * as THREE from 'three';
export class DebugVisualizer {
    constructor(scene) {
        this.scene = scene;
        this.debugObjects = {};
        this.sphereGeometry = new THREE.SphereGeometry(0.025, 16, 8);
        this.sphereMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        this.visible = true;
    }

    handlePositionsMessage(positionData) {
        this.updatePositions(positionData);
    }
    
    updatePositions(debugPositions) {
        if (!debugPositions || !this.visible) return;
        
        debugPositions.forEach((pos) => {
            // Check if position already exists
            if (this.debugObjects[pos.name]) {
                this.debugObjects[pos.name].position.set(
                    pos.position.x * 0.01, 
                    pos.position.y * 0.01, 
                    pos.position.z * 0.01
                );
            } else {
                // Create new debug object
                const sphere = new THREE.Mesh(this.sphereGeometry, this.sphereMaterial);
                sphere.name = pos.name;
                sphere.position.set(
                    pos.position.x * 0.01, 
                    pos.position.y * 0.01, 
                    pos.position.z * 0.01
                );
                this.scene.add(sphere);
                this.debugObjects[pos.name] = sphere;
            }
        });
    }
    
    setVisible(visible) {
        this.visible = visible;
        Object.values(this.debugObjects).forEach(obj => {
            obj.visible = visible;
        });
    }
    
    toggleVisibility() {
        this.setVisible(!this.visible);
        return this.visible;
    }
    
    clear() {
        Object.values(this.debugObjects).forEach(obj => {
            this.scene.remove(obj);
        });
        this.debugObjects = {};
    }
}