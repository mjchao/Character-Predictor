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
