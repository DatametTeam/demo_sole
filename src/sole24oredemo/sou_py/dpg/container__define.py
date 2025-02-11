import gc


class Container(object):
    """
    A container class for storing and managing a list of objects.

    This class provides methods to add, retrieve, and remove objects from an internal list. Objects can be of any type.
    The `get` method allows retrieving either all stored objects or only those of a specific type.

    Attributes:
        lst (list): The internal list where objects are stored.

    Methods:
        __init__(self, lst=[]): Initializes the container with an optional list of objects.
        add(self, new): Adds a new object to the container.
        get(self, classtype=None): Retrieves objects from the container. If `classtype` is provided,
                                   only objects of that type are returned.
        remove(self, obj): Removes an object from the container.
    """

    def __init__(self, lst=[]):
        """
        Initializes the Container with an optional list of objects.

        Args:
            lst (list, optional): A list of initial objects to store in the container. Defaults to an empty list.
        """
        self.lst = lst

    def add(self, new):
        """
        Adds a new object to the container.

        Args:
            new: The object to be added to the container.
        """
        self.lst.append(new)

    def get(self, classtype=None):
        """
        Retrieves objects from the container. If `classtype` is specified, only objects of that type are returned.

        Args:
            classtype (type, optional): The type of objects to retrieve. If None, all objects are returned. Defaults to None.

        Returns:
            list: A list of objects. The list contains either all objects or only those of the specified type.
        """
        """ 
        #Codice ottimizzato
        if classtype is None:
            return self.lst
        sublst = [obj for obj in self.lst if isinstance(obj, classtype)]
        return sublst
        """
        if classtype is None:
            # questa lista è sbagliata quando viene ritornata!
            return self.lst
        sublst = []
        for obj in self.lst:
            if isinstance(obj, classtype):
                sublst.append(obj)
            #
        #
        return sublst

    def remove(self, obj):
        """
        Removes an object from the container.

        Args:
            obj: The object to be removed from the container.

        Returns:
            None
        """
        if obj in self.lst:
            self.lst.remove(obj)
            del obj
        return

    def Cleanup(self, obj):
        """
        Releases the specified object and performs garbage collection.

        Parameters:
            obj:
                The object to be deleted and cleaned up.
        """
        if obj is not None:
            del obj
            gc.collect()
