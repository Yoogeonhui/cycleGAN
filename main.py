import tensorflow as tf
import numpy as np

from model import Model
from preprocess import LoadBatch

options = {
    'batchDirectoryX': './data/X',
    'batchDirectoryY': './data/Y',
    'learningRate' : 0.001,
    'checkPointName': 'cycleGAN.ckpt',
    'batchSize' : 100,
    'saveSize': 5,
}

if __name__ == '__main__':
    model = Model()
    GLoss, DLoss = model.getLoss()
    # save file check

    saver = tf.train.Saver()

    varList = tf.trainable_variables()
    GvarList = [var for var in varList if "GEN" in var.name]
    DvarList = [var for var in varList if "Dis" in var.name]
    adamOptimizer = tf.train.AdamOptimizer(learning_rate = options['learningRate'])
    print('GVARS', GvarList)
    print('DVARS', DvarList)
    trainG = adamOptimizer.minimize(GLoss, var_list = GvarList)
    trainD = adamOptimizer.minimize(DLoss, var_list = DvarList)

    global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    updated_step = tf.placeholder(tf.int32)
    assign_global_step = tf.assign(global_step_tensor, updated_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        tf.train.global_step(sess, global_step_tensor)

        if tf.train.checkpoint_exists(options['checkPointName']):
            saver.restore(sess,tf.train.latest_checkpoint(options['checkPointName']))
            global_step_tensor = tf.train.get_global_step()

        current_global_step = sess.run(global_step_tensor)

        loadbatchX = LoadBatch(options['batchDirectoryX'], step_num=current_global_step, batch_size=options['batchSize'])
        loadbatchY = LoadBatch(options['batchDirectoryY'], step_num=current_global_step, batch_size=options['batchSize'])

        while loadbatchX.getEpoch() < 50 or loadbatchY.getEpoch() < 50 :
            inputX = loadbatchX.getBatch()
            inputY = loadbatchY.getBatch()
            feed_dict = {model.inputX: inputX, model.inputY: inputY}
            DLossResult, _ = sess.run([DLoss, trainD], feed_dict)
            GLossResult, _ = sess.run([GLoss, trainG], feed_dict)
            print('DLoss: ',DLossResult,' GLoss: ',GLossResult)
            # 현재 step num은 loadBatch쪽 기준
            nowStep = loadbatchX.getStep()
            sess.run(assign_global_step, {updated_step: nowStep})

            if nowStep%(options['batchSize']*options['saveSize'])==0:
                saver.save(sess, options['checkPointName'], global_step = global_step_tensor)



