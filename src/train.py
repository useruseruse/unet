import torch 
from model.dice_loss import dice_loss


def train(model, optimizer, train_loader,  num_epochs=20):
    for epoch in range(num_epochs): 
        for batch_idx, (image1, image2) in enumerate(train_loader):
            ## for testing 
            output_1 = model(image1)        
            output_2 = model(image2)

            # Compute loss
            loss = dice_loss(output_1, output_2)  # Or use image2, depending on your setup

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # epoch 10, 20, 30 당 checkpoint 설정
        if(epoch % 10 == 0):
            torch.save({
                'epoch': epoch,
                'batch_idx': batch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'model': model,
            }, './model_cache/model_checkpoint_{}.pth'.format(epoch))
                        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss}')
    

def test(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for image_1, image_2  in test_loader:
            output_1 = model(image_1)
            output_2 = model(image_2)
            loss = dice_loss(output_1, output_2)
            # Here, you can print out loss, or store it for analysis
            print("loss", loss)

