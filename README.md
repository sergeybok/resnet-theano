# resnet-theano
(very) simple implementation of resnet architecture in theano in order to load the trained model for ResNet 101 for use with transfer learning

 Credits:
 1) Kaiming He et al for the original idea/architecture and the pretrained model(s) https://github.com/KaimingHe/deep-residual-networks
         
2) And the guys who wrote tensorpack for writing that script that turns a .caffemodel file into npy file, https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet


HUGE Disclaimer: I can't really tell if this is working exactly right because when I try to classify images I get an index as an output but I don't really know where the key is for this index (label thats' associated with the index). Tensorpack has synset_words.txt file however that file uses some weird encoding (e.g. n02105641) which I don't really understand and if I assume that the label is a line number of that file then it's for sure not working. However I believe it to be working on the basis of me inputing 5 different pictures of german shepherds and it always returning 904, so there's that. 


Essentially to get this working all you need to do is download the resnet101 caffemodel (deploy version) from the Kaiming He's repository on github. Then follow the instructions of tensorpack to convert that to an npy file and name it ResNet101.npy (I have that hardcoded should be pretty simple to change if you want though).


I did this with a batch of one (deploy version of the .caffemodel file) of resnet mostly because a batched version probably wouldn't build on my computer due to memory issues. However the code might work with a batched model as well since I don't remember the number of batches ever being explicit. Try it and let me know.

Also another note is that I didn't make any flags for building a brand new model because I don't intend to do that ever, however that shouldn't be too hard to figure out---simply change all the weight initialization to np.random.uniform() instead of the load_params[..].


To use it I usually run it like this:

         python -i resnet.py 

And then do 

         import cv2
         im = cv2.imread(name of image)
         im = np.asarray(im,dtype='float32')
         im = im / 255 # normalize, this is actually a step I'm not too sure about bc tensorpack does the normalization weirdly
         im = cv2.resize(im,(224,224)) # if image isn't a square I suggest cropping before this
         im = im.reshape((1,3,224,224))
         prediction = predict(im)
         
















