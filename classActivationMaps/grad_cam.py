from fastai.vision.all import *
import cv2

class Hook():
    """
    A function to hook forward pass outputs
    """
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_function)
    def hook_function(self, module, ip, op):
        self.stored = op.detach().clone()
    def __enter__(self, *args):
        return self
    def __exit__(self, *args):
        self.hook.remove()

class BackwardHook():
    """
    A function to keep track of backpropagated gradients
    """
    def __init__(self, module):
        self.hook = module.register_backward_hook(self.hook_function)
    def hook_function(self, module, ip, op):
        # The output is returned as a tuple grad of weights and grad of biases
        # We wanna extract grad of weights
        self.stored = op[0].detach().clone()
    def __enter__(self, *args):
        return self
    def __exit__(self, *args):
        self.hook.remove()
        

def plot_fig(ax, original_image, gradient_image, title = ""):
    """
    Given an image and it's corresponding heatmap representation, overlay heatmap with 0.5 transparency
    on top of the original input and present the same on a plt axis
    """
    ax.imshow(original_image)
    ax.imshow(gradient_image, alpha = 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def create_grad_cam_visualization(learn, pth):
    
    # Create an image and get it in a batch form
    img = PILImage.create(pth)
    x, = first(learn.dls.test_dl([img]))
    
    # Hook the output of the body for backward pass
    with BackwardHook(learn.model[0]) as hookg:
    
        # Get the number of classes in the output
        classes = learn.dls.vocab
        gradient_images = []
        confidence_scores = []

        for cls_ in range(len(classes)):
            
            # Hook the forward pass outputs
            with Hook(learn.model[0]) as hook:
                output = learn.model.eval()(x.cuda())
                confidence_scores.append(output[0].softmax(dim = 0)[cls_].item())
                act = hook.stored

            output[0, cls_].backward()
            grad = hookg.stored

            # Feature Map Weights
            feature_map_weights = grad.mean(axis = (-1, -2), keepdim = True)

            # Weighted Sum of backpropagated weights and the feature map activations
            feature_map_weighted_sum = (feature_map_weights * act).sum(axis = (0 , 1)).cpu()

            # Upsample the image to image dimension
            grad_cam_img = cv2.resize(feature_map_weighted_sum.numpy(), (224, 224))

            # Apply ReLU
            grad_cam_img = np.maximum(grad_cam_img, 0)

            # Normalize the grad cam image
            heatmap = (grad_cam_img - grad_cam_img.min()) / (grad_cam_img.max() - grad_cam_img.min())

            # Create colormap using the above heatmap
            grad_cam_img = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
            gradient_images.append(grad_cam_img)

        fig, ax = plt.subplots(1, len(classes), figsize = (10, 10))
        input_image = img.resize((224, 224))
        for gi, a, c, cs in zip(gradient_images, ax, classes, confidence_scores):
            plot_fig(a, input_image, gi, f"{c} Probability: {cs:.4f}")