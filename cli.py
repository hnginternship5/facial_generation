import click
#import genblack
import genwhite
import os
import tensorflow as tf

@click.command()
@click.argument('age')
@click.argument('region')
@click.argument('sex')
def main(age,region,sex):
    data_dir='/images'
    # Image configuration
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    
    batch_size = 16
    z_dim = 100
    learning_rate = 0.0002
    beta1 = 0.5
    epochs = 2
    if region=='black':
        pass
        #genblack.train(epochs, batch_size, z_dim, learning_rate, beta1, get_batches, shape)
    elif region=='white':
        data_files = genwhite.glob(os.path.join(data_dir, 'white/*.jpg'))
        shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, 3
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        with tf.Graph().as_default():
            genwhite.train(epochs, batch_size, z_dim, learning_rate, beta1,shape)
    else:
        click.echo("Enter either black or white as region")

if __name__=='__main__':
    main()
    
