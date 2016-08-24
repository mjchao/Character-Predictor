'''
Created on Aug 23, 2016

@author: mjchao
'''


class CharToIdDictionary(object):
    """Maps characters to integer IDs.
    """

    def __init__(self):
        pass

    def GetId(self, char):
        """Converts the given ASCII character to an integer ID.

        Args:
            char: (char) A character to convert to an ID.
        """
        return ord(char)

    def Size(self):
        """Gets the number of characters recognized by the dictionary
        
        Returns:
            size: (int) The number of characters recognized by the dictionary.
                The integer IDs will be from 0, 1, ..., size-1.
        """
        return 128
