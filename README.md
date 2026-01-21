# AttentionFusionNet
Fusion of pan and mul
AttentionFusionNet (this improved model):
Mechanism: CBAM (Convolitional Blockation Module) module is embedded after the residual block.
Innovation:
SpatialAttention: Generate a "heat map", tell the network where the edge is and where the texture is, and let the network enhance the details only where necessary.
ChannelAttention: automatically learn the importance weights of 8 bands, prevent the noise of a certain band from affecting the whole and reduce spectral distortion.
Evaluation: "The adaptive calibration of features has been realized, and the network has become smart, knowing' where to look' and' what to look at'."
