# WaterCNN Model Architecture

```mermaid
graph LR
    WaterIn["Water Input<br/>(B, 1, 3, 3)"] 
    TerrainIn["Terrain Input<br/>(B, 1, 3, 3)"]
    
    WaterIn --> WaterConv["Conv2d<br/>1→8, 3×3"]
    WaterConv --> WaterReLU["ReLU"]
    WaterReLU --> WaterOut["(B, 8, 1, 1)"]
    
    WaterIn --> CombConcat["Concatenate"]
    TerrainIn --> CombConcat
    CombConcat --> Comb1x1["Conv2d<br/>2→8, 1×1"]
    Comb1x1 --> CombReLU1["ReLU"]
    CombReLU1 --> Comb3x3["Conv2d<br/>8→16, 3×3"]
    Comb3x3 --> CombReLU2["ReLU"]
    CombReLU2 --> CombOut["(B, 16, 1, 1)"]

    TerrainIn --> TerrainConv["Conv2d<br/>1→8, 3×3"]
    TerrainConv --> TerrainReLU["ReLU"]
    TerrainReLU --> TerrainOut["(B, 8, 1, 1)"]
    
    
    
    WaterOut --> FlatConcat["Concatenate<br/>(B, 32)"]
    TerrainOut --> FlatConcat
    CombOut --> FlatConcat
    
    FlatConcat --> FC1["Linear<br/>32→16"]
    FC1 --> FCRELU["ReLU"]
    FCRELU --> FC2["Linear<br/>16→1"]
    FC2 --> Output["Output<br/>(B,)"]
    
    style WaterIn fill:#e1f5ff
    style TerrainIn fill:#f3e5f5
    style WaterOut fill:#e1f5ff
    style TerrainOut fill:#f3e5f5
    style CombOut fill:#fff3e0
    style Output fill:#c8e6c9
```

## Model Description

The **WaterCNN** model is a multi-branch convolutional neural network designed to predict water values at the next time step given 3×3 patches of water and terrain data.

### Architecture Details

**Input:**
- Water patches: (B, 1, 3, 3)
- Terrain patches: (B, 1, 3, 3)

**Three Processing Branches:**

1. **Water Branch** (Blue)
   - Conv2d(1→8, 3×3) → ReLU
   - Output: (B, 8, 1, 1)

2. **Terrain Branch** (Purple)
   - Conv2d(1→8, 3×3) → ReLU
   - Output: (B, 8, 1, 1)

3. **Combined Branch** (Orange)
   - Concatenate water & terrain → (B, 2, 3, 3)
   - Conv2d(2→8, 1×1) → ReLU → (B, 8, 3, 3)
   - Conv2d(8→16, 3×3) → ReLU → (B, 16, 1, 1)

**Head (Green):**
- Concatenate all three branches → (B, 32)
- Linear(32→16) → ReLU
- Linear(16→1)

**Output:** Single scalar value (B,)
