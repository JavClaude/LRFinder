from keras.callbacks import LambdaCallback
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np


class LRFinder:
    def __init__(self, model):
        '''
        Description
        -----------
        Donne de l'insight sur le Learning Rate:
        Pour chaque batch, le Learning Rate Varie exponentiellement entre deux bornes : min_Lr, max_Lr

        Modele.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        Lrfinder = LRFinder(Modele)
        Lrfinder.find_sample_fit(X_train, y_train, min_lr = 0.0001, max_lr = 2, X_test, y_test, batch_size = 512, epochs = 1)
        Lrfinder.plot()

        Parametres
        ----------
        model: Keras model
        
        '''
        self.model = model
        self.losses = []
        self.acc = []
        self.lrs = []
        self.best_loss = 1e5
        self.best_acc = 0
        self.validation_set = None

    def on_batch_end(self, batch, logs):
        '''
        Description
        -----------
        Methode appelee par le CallBack du modele qui permet l'actualisation du learning rate pour le batch suivant

        Parametres
        ----------
        batch, logs
        
        '''
        # LR pour le batch donne
        lr = K.get_value(self.model.optimizer.lr)

        self.lrs.append(lr)

        # Logs du callback
        loss = logs['loss']
        acc = logs['accuracy']

        # Mise a jour des metrics
        self.losses.append(loss)
        self.acc.append(acc)

        if loss < self.best_loss:
            self.best_loss = loss

        if acc > self.best_acc:
            self.best_acc = acc

        # Augmentation du LR pour le batch suivant
        lr *= self.lr_exp
        K.set_value(self.model.optimizer.lr, lr)

    def find_sample_fit(self, X_train, y_train, min_lr, max_lr,
                        X_test=None, y_test=None, batch_size=64, epochs=1):
        '''
        Description
        -----------
        Entraine le modele:
            - Variation exponentielle du LR entre les deux bornes (min, max)
            
        Parametres
        ----------
            - X_train, y_train: Array,
            - X_test, y_test: Array,
            - min_lr, max_lr: Float, bornes min et max entre lesquelles faire varier exponentiellement le learning rate
            - batch_size: Int, taille du batch pour l'apprentissage
            - epochs: Int, nombre d'iteration pour l'entrainement

        Return
        ------
        '''

        self.n_samples = X_train.shape[0]

        self.batch_size = batch_size

        self.n_batches = epochs * self.n_samples / batch_size

        self.lr_exp = (float(max_lr) / float(min_lr)) ** (float(1) / float(self.n_batches))

        initial_lr = K.get_value(self.model.optimizer.lr)

        K.set_value(self.model.optimizer.lr, min_lr)

        # LambdaCallback
        #     - on_batch_end requiert deux arguments: batch, logs
        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        if X_test is not None and y_test is not None:
            self.validation_set = (X_test, y_test)
            self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                           validation_data=self.validation_set, callbacks=[callback])
        else:
            self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[callback])

        K.set_value(self.model.optimizer.lr, initial_lr)

    def derivatives_loss(self, sma=int):
        '''
        Description
        -----------
        Calcul le taux de variation de la loss avec lissage ()


        Parametres
        ----------
            - sma: Int, Nombre de batch a selectionner pour le lissage de la fonction de perte

        Return
        ------
        loss_derivates: List, Liste contenant les derivees de la fonction de pertes
        
        '''
        
        loss_derivatives = [0] * (sma + 1)
        for i in range(1 + sma, len(self.lrs)):
            derivative = (self.losses[i] - self.losses[i - sma]) / sma
            loss_derivatives.append(derivative)

        return loss_derivatives

    def plot(self, n_skip=5, derivative=False, sma=10):
        '''
        Description
        -----------
        Representation graphique de la fonction de perte/LearningRate

        Parametres
        ----------
            - sma: Int, Nombre de batch a selectionner pour le lissage de la fonction de perte
            - n_skip: Int, Nombre de batch a sauter (LRs extremes)

        Return
        ------
        '''
        
        fig = plt.figure(figsize=(12, 6))

        lrs_temp = self.lrs[n_skip:-n_skip]
        loss_temp = self.losses[n_skip:-n_skip]
        acc_temp = self.acc[n_skip:-n_skip]
        
        if derivative:
            self.derivatives = self.derivatives_loss(sma=sma)
            derivatives_loss_temp = self.derivatives[n_skip:-n_skip]

            plt.plot(lrs_temp, derivatives_loss_temp, zorder = 1)
            plt.xscale('log')
            plt.scatter(lrs_temp[np.argmin(derivatives_loss_temp)], derivatives_loss_temp[np.argmin(derivatives_loss_temp)], c = 'r', s = 50, zorder = 2)
            plt.title('dLoss')
            plt.show()
            print("MinLoss: {}\nLR: {}" .format(loss_temp[np.argmin(loss_temp)], lrs_temp[np.argmin(loss_temp)]))
            print("------------")
            print("MaxAcc: {}\nLR: {}" .format(acc_temp[np.argmax(acc_temp)], lrs_temp[np.argmax(acc_temp)]))

        else:
            plt.plot(lrs_temp, loss_temp, zorder = 1)
            plt.xscale('log')
            plt.scatter(lrs_temp[np.argmin(loss_temp)], loss_temp[np.argmin(loss_temp)], c = 'r', s = 50, zorder = 2)
            plt.title('Loss')
            plt.show()
            print("MinLoss: {}\nLR: {}" .format(loss_temp[np.argmin(loss_temp)], lrs_temp[np.argmin(loss_temp)]))
            print("------------")
            print("MaxAcc: {}\nLR: {}" .format(acc_temp[np.argmax(acc_temp)], lrs_temp[np.argmax(acc_temp)]))

