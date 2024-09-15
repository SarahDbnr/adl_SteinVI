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


def print_adult_income_dataset_info():
    """
    Prints a detailed description of the Adult Income dataset.
    """
    print("Dataset Description")
    print("\t• Number of Instances: 48,842 (32,561 training, 16,281 test)")
    print("\t• Number of Features: 14")
    print("\t• Number of Classes: 2")
    print("\t• Features:")
    print("\t\t• Age: Continuous.")
    print("\t\t• Workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, etc.")
    print("\t\t• Education: Bachelors, Some-college, 11th, HS-grad, Prof-school, etc.")
    print("\t\t• Marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed.")
    print("\t\t• Occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, etc.")
    print("\t\t• Relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.")
    print("\t\t• Race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.")
    print("\t\t• Sex: Female, Male.")
    print("\t\t• Capital-gain: Continuous.")
    print("\t\t• Capital-loss: Continuous.")
    print("\t\t• Hours-per-week: Continuous.")
    print("\t\t• Native-country: United-States, Cambodia, England, Puerto-Rico, Canada, etc.")
    print("\t• Target Classes:")
    print("\t\t• <=50K: Individuals earning less than or equal to $50,000.")
    print("\t\t• >50K: Individuals earning more than $50,000.")


def print_california_housing_dataset_info():
    """
    Prints a detailed description of the California Housing dataset.
    """
    print("Dataset Description")
    print("\t• Number of Instances: 20,640")
    print("\t• Number of Features: 8")
    print("\t• Features:")
    print("\t\t• MedInc: Median income in block group. Continuous. Range: [0.4999, 15.0001]")
    print("\t\t• HouseAge: Median house age in block group. Continuous. Range: [1.0, 52.0]")
    print("\t\t• AveRooms: Average number of rooms per household. Continuous. Range: [2.0, 141.909]")
    print("\t\t• AveBedrms: Average number of bedrooms per household. Continuous. Range: [1.0, 34.066]")
    print("\t\t• Population: Block group population. Continuous. Range: [3.0, 35682.0]")
    print("\t\t• AveOccup: Average number of household members. Continuous. Range: [0.692, 1243.333]")
    print("\t\t• Latitude: Block group latitude. Continuous. Range: [32.54, 41.95]")
    print("\t\t• Longitude: Block group longitude. Continuous. Range: [-124.35, -114.31]")
    print("\t• Target:")
    print("\t\t• MedHouseVal: Median house value in block group (in $100,000s). Continuous. Range: [0.14999, 5.00001]")


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


def print_wine_quality_dataset_info():
    """
    Prints a detailed description of the Wine Quality dataset.
    """
    print("Dataset Description")
    print("\t• Number of Instances: 178")
    print("\t• Number of Features: 13")
    print("\t• Features:")
    print("\t\t• Alcohol: Alcohol content in wine. Continuous. Range: [11.03, 14.83]")
    print("\t\t• Malic acid: Malic acid content. Continuous. Range: [0.74, 5.80]")
    print("\t\t• Ash: Ash content. Continuous. Range: [1.36, 3.23]")
    print("\t\t• Alcalinity of ash: Alkaline content of ash. Continuous. Range: [10.6, 30.0]")
    print("\t\t• Magnesium: Magnesium content. Continuous. Range: [70.0, 162.0]")
    print("\t\t• Total phenols: Total phenolic content. Continuous. Range: [0.98, 3.88]")
    print("\t\t• Flavanoids: Flavonoid content. Continuous. Range: [0.34, 5.08]")
    print("\t\t• Nonflavanoid phenols: Non-flavonoid phenolic content. Continuous. Range: [0.13, 0.66]")
    print("\t\t• Proanthocyanins: Proanthocyanin content. Continuous. Range: [0.41, 3.58]")
    print("\t\t• Color intensity: Intensity of color in wine. Continuous. Range: [1.28, 13.0]")
    print("\t\t• Hue: Hue of wine. Continuous. Range: [0.48, 1.71]")
    print("\t\t• OD280/OD315 of diluted wines: Dilution in wine. Continuous. Range: [1.27, 4.00]")
    print("\t\t• Proline: Proline content. Continuous. Range: [278.0, 1680.0]")
    print("\t• Target:")
    print("\t\t• Wine Class: Classification of wine into one of three categories (0, 1, 2). Discrete.")



def print_bike_sharing_dataset_info():
    """
    Prints a detailed description of the Bike Sharing dataset.
    """
    print("Dataset Description")
    print("\t• Name: Bike Sharing Dataset")
    print("\t• Number of Instances: 17,379")
    print("\t• Number of Features: 16")
    print("\t• Features:")
    print("\t\t• season: Season (1:springer, 2:summer, 3:fall, 4:winter). Categorical.")
    print("\t\t• yr: Year (0: 2011, 1:2012). Categorical.")
    print("\t\t• mnth: Month (1 to 12). Categorical.")
    print("\t\t• hr: Hour (0 to 23). Categorical.")
    print("\t\t• holiday: Whether the day is a holiday or not. Binary.")
    print("\t\t• weekday: Day of the week (0 to 6). Categorical.")
    print("\t\t• workingday: Whether the day is a working day or not. Binary.")
    print("\t\t• weathersit: Weather situation (1: Clear, 2: Mist, 3: Light Snow, 4: Heavy Rain). Categorical.")
    print("\t\t• temp: Normalized temperature in Celsius. Continuous.")
    print("\t\t• atemp: Normalized feeling temperature in Celsius. Continuous.")
    print("\t\t• hum: Normalized humidity. Continuous.")
    print("\t\t• windspeed: Normalized wind speed. Continuous.")
    print("\t\t• casual: Number of casual users. Continuous.")
    print("\t\t• registered: Number of registered users. Continuous.")
    print("\t\t• cnt: Total number of rentals (casual + registered). Continuous.")
    
    print("\t• Target:")
    print("\t\t• cnt: Total number of bike rentals for a given hour.")