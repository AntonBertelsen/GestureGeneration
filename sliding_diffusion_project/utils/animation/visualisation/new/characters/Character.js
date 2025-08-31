import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { BVHLoader } from 'three/addons/loaders/BVHLoader.js'; // Add BVH loader

export class Character {
    
    constructor(name, scene, position = null, rotation = null, color = null) {
        this.name = name;
        this.modelPath = 'model.glb'; // Default model path, can be overridden
        this.bvhPath = 'default.bvh'; // Default BVH path, can be overridden
        this.scene = scene;
        this.model = null;
        this.skeleton = null;
        this.skeletonHelper = null;
        this.bvhData = null; // Add BVH data
        this.bvhSkeleton = null; // Add BVH skeleton
        this.bvhMixer = null; // Add BVH mixer
        this.boneContainer = new THREE.Group();
        this.debugPositions = [];
        this.visible = true;
        this.mixer = null;
        this.actions = {};
        this.bonesMap = {};
        this.originalBonePositions = {};
        this.showSkeleton = false;
        this.showMesh = true;
        
        // New properties for positioning and color
        this.position = position || new THREE.Vector3(0, 0, 0);
        this.rotation = rotation ? THREE.MathUtils.degToRad(rotation) : 0; // in radians
        this.color = color;
    }
    
    async load() {
        // Load both model and BVH in parallel
        await Promise.all([
            this.loadModel(),
            this.loadBVH()
        ]);
        
        // Set up connection between BVH and model bones
        this.connectBVHToModel();
        
        console.log(`Character ${this.name} loaded successfully`);
        return this;
    }
    
    async loadBVH() {
        return new Promise((resolve, reject) => {
            const bvhLoader = new BVHLoader();
            bvhLoader.load(this.bvhPath, (bvh) => {
                this.bvhData = bvh;
                this.bvhSkeleton = bvh.skeleton;
                
                // Scale the skeleton to match model size (1% of original size)
                this.bvhSkeleton.bones[0].scale.set(0.01, 0.01, 0.01);
                
                // Create BVH skeleton helper
                this.bvhSkeletonHelper = new THREE.SkeletonHelper(this.bvhSkeleton.bones[0]);
                this.bvhSkeletonHelper.skeleton = this.bvhSkeleton;
                this.bvhSkeletonHelper.visible = false;
                this.scene.add(this.bvhSkeletonHelper);
                
                // Add bone container to scene
                this.bvhBoneContainer = new THREE.Group();
                this.bvhBoneContainer.add(this.bvhSkeleton.bones[0]);
                this.scene.add(this.bvhBoneContainer);
                
                // Create animation mixer for BVH
                this.bvhMixer = new THREE.AnimationMixer(this.bvhSkeleton.bones[0]);
                
                // Store original BVH bone positions
                this.originalBVHPositions = {};
                this.bvhSkeleton.bones.forEach((bone) => {
                    this.originalBVHPositions[bone.name] = bone.position.clone();
                });
                
                console.log(`BVH skeleton loaded for ${this.name}`);
                resolve();
            }, undefined, reject);
        });
    }
    
    async loadModel() {
        return new Promise((resolve, reject) => {
            const gltfLoader = new GLTFLoader();
            gltfLoader.load(this.modelPath, (gltf) => {
                this.model = gltf.scene;
                
                // Apply color if specified
                if (this.color) {
                    this.setColor(this.color);
                }

                this.scene.add(this.model);
                
                // Map all bones
                this.mapBones();
                
                // Set up skeleton helper
                let skinnedMesh = null;
                this.model.traverse((object) => {
                    if (object.isSkinnedMesh) {
                        skinnedMesh = object;
                    }
                });
                
                if (skinnedMesh) {
                    this.skeletonHelper = new THREE.SkeletonHelper(skinnedMesh.skeleton.bones[0]);
                    this.skeletonHelper.visible = false;
                    this.scene.add(this.skeletonHelper);
                    console.log(`Skeleton helper set up for ${this.name}`);
                } else {
                    console.warn(`No skinned mesh found in model for ${this.name}, creating skeleton helper for entire model`);
                    this.skeletonHelper = new THREE.SkeletonHelper(this.model);
                    this.skeletonHelper.visible = false;
                    this.scene.add(this.skeletonHelper);
                }
                
                // Add bone container to scene
                this.scene.add(this.boneContainer);
                
                // Create animation mixer
                this.mixer = new THREE.AnimationMixer(this.model);
                
                // Store original bone positions
                this.saveBonePositions();
                
                resolve();
            }, undefined, reject);
        });
    }
    
    connectBVHToModel() {
        if (!this.bvhSkeleton || !this.model) {
            console.warn('Cannot connect BVH to model: missing skeleton or model');
            return;
        }
        
        console.log('Connecting BVH bones to model bones');
        
        // Match and connect BVH bones to GLB bones
        let matchedCount = 0;
        for (let i = 0; i < this.bvhSkeleton.bones.length; i++) {
            const bvhBone = this.bvhSkeleton.bones[i];
            const glbBone = this.bonesMap[bvhBone.name];
            
            if (glbBone) {
                matchedCount++;
                if (bvhBone.name !== 'ENDSITE') {
                    bvhBone.attach(glbBone);
                }
            }
        }
        
        console.log(`Successfully matched ${matchedCount} out of ${this.bvhSkeleton.bones.length} bones`);
    }
    
    mapBones() {
        // Create a map of all bones in the model for quick access
        this.model.traverse((object) => {
            if (object.isBone || object.type === 'Bone') {
                this.bonesMap[object.name] = object;
            }
        });
        
        console.log(`Found ${Object.keys(this.bonesMap).length} bones in model for ${this.name}`);
    }
    
    saveBonePositions() {
        Object.values(this.bonesMap).forEach((bone) => {
            this.originalBonePositions[bone.name] = bone.position.clone();
        });
    }
    
    update(delta) {
        // Update both mixers
        //if (this.mixer) {
        //    this.mixer.update(delta);
        //}
        
        //if (this.bvhMixer) {
        //    this.bvhMixer.update(delta);
        //}
        
        // Restore original bone positions (if needed)
        // Uncommenting this block would reset non-root bones to their original positions
        /*
        Object.entries(this.bonesMap).forEach(([name, bone]) => {
            if (name !== 'hip' && name !== 'root') { // Don't reset root bones
                const originalPosition = this.originalBonePositions[name];
                if (originalPosition) {
                    bone.position.copy(originalPosition);
                }
            }
        });
        
        // Also restore BVH bones if needed
        Object.entries(this.bvhSkeleton.bones).forEach((bone) => {
            if (bone.name !== 'body_world' && bone.name !== 'ENDSITE') {
                const originalPosition = this.originalBVHPositions[bone.name];
                if (originalPosition) {
                    bone.position.copy(originalPosition);
                }
            }
        });
        */
    }
    
    applyPoseData(poseData) {
        if (!this.bvhSkeleton) {
            console.warn('No BVH skeleton available to apply pose data');
            return;
        }
        
        Object.entries(poseData).forEach(([boneName, jointData]) => {
            // Find the bone in the BVH skeleton
            const bone = this.bvhSkeleton.bones.find(b => b.name === boneName);
            
            if (bone) {
                // Apply position (typically only for root bone)
                if (jointData.position) {
                    bone.position.set(
                        jointData.position.x * 0.01,
                        jointData.position.y * 0.01,
                        jointData.position.z * 0.01
                    );
                }
                
                // Apply rotation to bone
                if (jointData.eulerAngles) {
                    const euler = new THREE.Euler(
                        jointData.eulerAngles.x * Math.PI/180,
                        jointData.eulerAngles.y * Math.PI/180,
                        jointData.eulerAngles.z * Math.PI/180,
                        'ZXY'
                    );
                    
                    bone.setRotationFromEuler(euler);
                }
            } else {
                console.warn(`Bone ${boneName} not found in BVH skeleton`);
            }
        });
        // get the model's root bone
        const rootBone = this.bvhSkeleton.bones[0];
        // Update the position of the root bone
        if (rootBone) {
            rootBone.position.applyAxisAngle(new THREE.Vector3(0, 1, 0), this.rotation);
            rootBone.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), this.rotation);
            rootBone.position.add(this.position);
        } else {
            console.warn('Root bone not found in BVH skeleton');
        }

    }
    
    setDisplayMode(showMesh, showSkeleton) {
        this.showMesh = showMesh;
        this.showSkeleton = showSkeleton;
        
        // Update visibilities
        if (this.model) {
            this.model.visible = this.visible && this.showMesh;
        }
        
        if (this.skeletonHelper) {
            this.skeletonHelper.visible = this.visible && this.showSkeleton;
        }
        
        if (this.bvhSkeletonHelper) {
            this.bvhSkeletonHelper.visible = this.visible && this.showSkeleton;
        }
        
        console.log(`Display mode updated for ${this.name}: Mesh=${this.showMesh}, Skeleton=${this.showSkeleton}`);
    }

    setColor(color) {
        // Traverse all meshes and set their material color
        this.model.traverse((object) => {
            if (object.isMesh && object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(mat => {
                        if (mat.color) {
                            mat.color.set(color);
                            mat.needsUpdate = true;
                        }
                    });
                } else {
                    if (object.material.color) {
                        object.material.color.set(color);
                        object.material.needsUpdate = true;
                    }
                }
            }
        });
        this.model.traverse((object) => {
            if (object.isMesh) {
                console.log(`${object.name} material:`, object.material.type);
            }
        });
    }
}