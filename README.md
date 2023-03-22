## Classification of handwritten chess move records

The model recognizes written text using the [AlexNet](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) architecture.

You can test the model's performance at [ChessVisionLab.com](https://ChessVisionLab.com)


### Create environment for train and evaluate model
1. Clone chess-vision-lab repo
    ``` bash
    git clone https://github.com/JacekMaksymiuk/chess-vision-lab && cd chess-vision-lab
    ```
2. Prepare env: unzip dataset and create necessary folders 
    ``` bash
    make install
    ```

### Train and validate model
1. Run console in docker
    ``` bash
    make console
    ```
2. Start training in console
    ``` bash
    ../bin/train.sh
    ```

### Runs the classification for the given images of handwritten moves
1. Run console in docker
    ``` bash
    make console
    ```
2. Start training in console
    ``` bash
    ../bin/train.sh
    ```
