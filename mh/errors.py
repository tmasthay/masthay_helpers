def easy_handle(val: bool, err: Exception):
    if val:
        raise err


class NotEnoughResourcesError(Exception):
    pass


class ShapeError(Exception):
    def __init__(self, *, expected, data, field, msg=''):
        if not hasattr(data, 'shape'):
            raise ValueError(
                "Expected data to have a shape attribute, check data type"
                f" below\n    {type(data)=}"
            )
        self.data_shape = data.shape
        self.expected = expected
        self.field = field
        self.msg = msg
        super().__init__(
            f"Shape Mismatch!\n    Expected{expected} for {field}, got"
            f" {data.shape}\n{msg}"
        )
