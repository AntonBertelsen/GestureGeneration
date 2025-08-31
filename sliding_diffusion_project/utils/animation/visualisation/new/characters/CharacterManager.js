import * as THREE from 'three';
import { Character } from './Character.js';

export class CharacterManager {

    static DISPLAY_MODE = {
        MESH_ONLY: 'mesh_only',
        BOTH: 'both',
        SKELETON_ONLY: 'skeleton_only'
    };

    constructor(scene) {
        this.scene = scene;
        this.characters = {};
        this.sphereGeometry = new THREE.SphereGeometry(0.025, 16, 8);
        this.sphereMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });

        this.currentDisplayMode = CharacterManager.DISPLAY_MODE.MESH_ONLY;
    }

    async handleAddCharacterMessage(data) {
        try {
            const { name, position, rotation, color } = data;
            this.addCharacter(name, position, rotation, color);
            console.log(`Character ${name} added successfully.`);
        } catch (error) {
            console.error('Error handling add character message:', error);
        }
    }

    async handlePoseMessage(data) {
        try {
            const { pose, pose_target_character } = data;
            this.applyPoseDataToCharacter(pose, pose_target_character);
        } catch (error) {
            console.error('Error handling pose message:', error);
        }
    }
    
    async addCharacter(name, position = null, rotation = null, color = null) {
        // Check if a character with this name already exists
        if (this.characters[name]) {
            console.warn(`Character with name ${name} already exists. Choose a different name.`);
            return null;
        }
        
        const character = new Character(name, this.scene, position, rotation, color);
        await character.load();
        this.characters[name] = character;

        // Ensure the new character follows the current display mode
        this.setDisplayMode(this.currentDisplayMode);
        return character;
    }

    getCharacter(name) {
        return this.characters[name] || null;
    }
        
    update(delta) {
        Object.values(this.characters).forEach(character => {
            if (character.visible) {
                character.update(delta);
            }
        });
    }
    
    applyPoseDataToCharacter(poseData, characterName) {
        const character = this.getCharacter(characterName);
        if (!character) {
            console.warn(`Character ${characterName} not found.`);
            return;
        }
        character.applyPoseData(poseData);
    }
    
    getCharacterNames() {
        return Object.keys(this.characters);
    }
    
    getCurrentDisplayMode() {
        return this.currentDisplayMode;
    }
    
    // Replace individual toggle methods with this
    setDisplayMode(mode) {
        this.currentDisplayMode = mode;
        Object.values(this.characters).forEach(character => {
            character.setDisplayMode(
                mode === CharacterManager.DISPLAY_MODE.MESH_ONLY,
                mode === CharacterManager.DISPLAY_MODE.BOTH
            );
        });
    }
    
    // Add method to cycle through display modes
    cycleDisplayMode() {
        switch(this.currentDisplayMode) {
            case CharacterManager.DISPLAY_MODE.MESH_ONLY:
                this.setDisplayMode(CharacterManager.DISPLAY_MODE.BOTH);
                break;
            case CharacterManager.DISPLAY_MODE.BOTH:
                this.setDisplayMode(CharacterManager.DISPLAY_MODE.SKELETON_ONLY);
                break;
            case CharacterManager.DISPLAY_MODE.SKELETON_ONLY:
                this.setDisplayMode(CharacterManager.DISPLAY_MODE.MESH_ONLY);
                break;
        }
        
        return this.currentDisplayMode;
    }
}