import { BVHLoader } from 'three/addons/loaders/BVHLoader.js';

export class AnimationManager {
    async loadBVHAnimation(name, bvhPath) {
        return new Promise((resolve, reject) => {
            const bvhLoader = new BVHLoader();
            bvhLoader.load(bvhPath, (bvh) => {
                // Process the animation
                const clip = bvh.clip;
                
                // Scale down the root motion if needed
                clip.tracks.forEach((track) => {
                    if (track.name.endsWith('.position')) {
                        for (let i = 0; i < track.values.length; i++) {
                            track.values[i] *= 0.01; // Scale factor
                        }
                    }
                });
                
                this.animations[name] = {
                    clip,
                    skeleton: bvh.skeleton,
                    duration: clip.duration
                };
                
                resolve(this.animations[name]);
            }, undefined, reject);
        });
    }
}