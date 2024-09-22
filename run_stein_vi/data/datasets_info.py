def print_iris_dataset_info():
    """
    Prints a detailed description of the Iris dataset.
    """
    print("Dataset Description")
    print("\t• Number of Instances: 150")
    print("\t• Number of Features: 4")
    print("\t• Number of Classes: 3")
    print("\t• Features:")
    print("\t\t• Sepal Length: The length of the sepal (in centimeters).")
    print("\t\t• Sepal Width: The width of the sepal (in centimeters).")
    print("\t\t• Petal Length: The length of the petal (in centimeters).")
    print("\t\t• Petal Width: The width of the petal (in centimeters).")
    print("\t• Target Classes:")
    print("\t\t• Setosa: Iris-setosa")
    print("\t\t• Versicolour: Iris-versicolor")
    print("\t\t• Virginica: Iris-virginica")


def print_mnist_dataset_info():
    """
    Prints a detailed description of the MNIST dataset.
    """
    print("Dataset Description")
    print("\t• Number of Instances: 70,000 (60,000 training, 10,000 test)")
    print("\t• Number of Features: 784 (28x28 pixels)")
    print("\t• Number of Classes: 10")
    print("\t• Features:")
    print("\t\t• Each instance is a 28x28 grayscale image of a handwritten digit.")
    print("\t• Target Classes:")
    print("\t\t• 0: Digit '0'")
    print("\t\t• 1: Digit '1'")
    print("\t\t• 2: Digit '2'")
    print("\t\t• 3: Digit '3'")
    print("\t\t• 4: Digit '4'")
    print("\t\t• 5: Digit '5'")
    print("\t\t• 6: Digit '6'")
    print("\t\t• 7: Digit '7'")
    print("\t\t• 8: Digit '8'")
    print("\t\t• 9: Digit '9'")


def print_fashion_mnist_dataset_info():
    """
    Prints a detailed description of the Fashion MNIST dataset.
    """
    print("Dataset Description")
    print("\t• Number of Instances: 70,000 (60,000 training, 10,000 test)")
    print("\t• Number of Features: 784 (28x28 pixels)")
    print("\t• Number of Classes: 10")
    print("\t• Features:")
    print("\t\t• Each instance is a 28x28 grayscale image of a fashion item.")
    print("\t• Target Classes:")
    print("\t\t• 0: T-shirt/top")
    print("\t\t• 1: Trouser")
    print("\t\t• 2: Pullover")
    print("\t\t• 3: Dress")
    print("\t\t• 4: Coat")
    print("\t\t• 5: Sandal")
    print("\t\t• 6: Shirt")
    print("\t\t• 7: Sneaker")
    print("\t\t• 8: Bag")
    print("\t\t• 9: Ankle boot")


def print_cifar10_dataset_info():
    """
    Prints a detailed description of the CIFAR10 dataset.
    """
    print("Dataset Description")
    print("\t• Number of Instances: 60,000 (50,000 training, 10,000 test)")
    print("\t• Number of Features: 3,072 (32x32 pixels, 3 color channels)")
    print("\t• Number of Classes: 10")
    print("\t• Features:")
    print("\t\t• Each instance is a 32x32 color image.")
    print("\t• Target Classes:")
    print("\t\t• 0: Airplane")
    print("\t\t• 1: Automobile")
    print("\t\t• 2: Bird")
    print("\t\t• 3: Cat")
    print("\t\t• 4: Deer")
    print("\t\t• 5: Dog")
    print("\t\t• 6: Frog")
    print("\t\t• 7: Horse")
    print("\t\t• 8: Ship")
    print("\t\t• 9: Truck")


def print_20_newsgroups_dataset_info():
    """
    Prints a detailed description of the Newsgroup_20 dataset.
    """
    print("Dataset Description")
    print("\t• Number of Instances: 18,846 (11,314 training, 7,532 test)")
    print("\t• Number of Features: Variable (depending on text vectorization, standardly here 2000)")
    print("\t• Number of Classes: 20")
    print("\t• Features:")
    print("\t\t• Each instance is a text document belonging to one of 20 different newsgroups.")
    print("\t• Target Classes (Newsgroups):")
    print("\t\t• alt.atheism")
    print("\t\t• comp.graphics")
    print("\t\t• comp.os.ms-windows.misc")
    print("\t\t• comp.sys.ibm.pc.hardware")
    print("\t\t• comp.sys.mac.hardware")
    print("\t\t• comp.windows.x")
    print("\t\t• misc.forsale")
    print("\t\t• rec.autos")
    print("\t\t• rec.motorcycles")
    print("\t\t• rec.sport.baseball")
    print("\t\t• rec.sport.hockey")
    print("\t\t• sci.crypt")
    print("\t\t• sci.electronics")
    print("\t\t• sci.med")
    print("\t\t• sci.space")
    print("\t\t• soc.religion.christian")
    print("\t\t• talk.politics.guns")
    print("\t\t• talk.politics.mideast")
    print("\t\t• talk.politics.misc")
    print("\t\t• talk.religion.misc")


def print_diabetes_dataset_info():
    """
    Prints a detailed description of the Diabetes dataset.
    """
    print("Dataset Description")
    print("\t• Number of Instances: 442")
    print("\t• Number of Features: 10")
    print("\t• Features:")
    print("\t\t• Age: Normalized. Continuous. Range: [-0.10722563, 0.11072668]")
    print("\t\t• Sex: Normalized. Continuous. Range: [-0.04464164, 0.05068012]")
    print("\t\t• BMI: Body mass index. Normalized. Continuous. Range: [-0.09027529, 0.17055523]")
    print("\t\t• BP: Average blood pressure. Normalized. Continuous. Range: [-0.11239936, 0.13204451]")
    print("\t\t• S1: T-cells (a type of white blood cells). Normalized. Continuous. Range: [-0.12609759, 0.15391371]")
    print("\t\t• S2: Low-density lipoproteins. Normalized. Continuous. Range: [-0.11561307, 0.19878799]")
    print("\t\t• S3: High-density lipoproteins. Normalized. Continuous. Range: [-0.10230705, 0.18117906]")
    print("\t\t• S4: Thyroid stimulating hormone. Normalized. Continuous. Range: [-0.0763945, 0.18523444]")
    print("\t\t• S5: Lamotrigine. Normalized. Continuous. Range: [-0.1264712, 0.13359898]")
    print("\t\t• S6: Blood sugar level. Normalized. Continuous. Range: [-0.13776723, 0.13561183]")
    print("\t• Target:")
    print("\t\t• Disease progression one year after baseline. Continuous. Range: [25.0, 346.0]")