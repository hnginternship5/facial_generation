import click
import os
import generateface
import tensorflow as tf

@click.command()
@click.argument('age')
@click.argument('region')
@click.argument('gender')
def main(age,region,gender):
    data_dir='/images'
    # Image configuration
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    
    batch_size = 16
    z_dim = 100
    learning_rate = 0.0002
    beta1 = 0.5
    epochs = 2
    genders=['male','female']
    regions=['black','white']
    if (region in regions) and (gender in genders):
        data_files = generateface.glob(os.path.join(data_dir, '%s/%s/*.*' %(region,gender)))
        shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, 3
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        with tf.Graph().as_default():
            generateface.train(epochs, batch_size, z_dim, learning_rate, beta1,shape)
    else :
        click.echo("Enter either black or white as region")
        click.echo('Enter either male or female as gender')
        click.echo('E.g 24 black female')


if __name__=='__main__':
    main()
    
