# import torch
# from torch.utils.checkpoint import checkpoint
# from health_multimodal.image.model.resnet import ResNetHIML

# # This is a simplified example - you'll need to adapt to your actual ResNet implementation
# def forward_with_checkpointing(self, x):
#     """Forward method with gradient checkpointing for ResNet blocks."""
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.relu(x)
#     x = self.maxpool(x)
    
#     # Apply checkpointing to each layer
#     if getattr(self, 'gradient_checkpointing', False):
#         x = checkpoint(lambda x: self._forward_layer(self.layer1, x), x)
#         x = checkpoint(lambda x: self._forward_layer(self.layer2, x), x)
#         x = checkpoint(lambda x: self._forward_layer(self.layer3, x), x)
#         x = checkpoint(lambda x: self._forward_layer(self.layer4, x), x)
#     else:
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
    
#     x = self.avgpool(x)
#     x = torch.flatten(x, 1)
#     x = self.fc(x)
    
#     return x

# def _forward_layer(self, layer, x):
#     """Helper function for checkpointing."""
#     return layer(x) 

# # Add this import at the top of your main.py or wherever you create your model
# from health_multimodal.image.model.resnet_modified import patch_resnet_with_checkpointing

# # Call this before creating your model
# patch_resnet_with_checkpointing()

# # Then create your model as usual
# visual_model = get_biovil_t_image_encoder() 