from databroker import Broker
#from databroker.core import register_builtin_handlers
from filestore.fs import FileStore

db_old = Broker.named('hxn_old')
db_new = Broker.named('hxn')



from hxntools.handlers.xspress3 import Xspress3HDF5Handler
from hxntools.handlers.timepix import TimepixHDF5Handler

#register_builtin_handlers(db_new.reg)

#db_new.reg.register_handler(Xspress3HDF5Handler.HANDLER_NAME,
#                            Xspress3HDF5Handler)
db_new.reg.register_handler(TimepixHDF5Handler._handler_name,
                            TimepixHDF5Handler, overwrite=True)


#register_builtin_handlers(db_old.reg)
#db_old.reg.register_handler(Xspress3HDF5Handler.HANDLER_NAME,
#                            Xspress3HDF5Handler)
db_old.reg.register_handler(TimepixHDF5Handler._handler_name,
                            TimepixHDF5Handler, overwrite=True)


# wrapper for two databases
class Broker_New(Broker):

    def __getitem__(self, key):
        try:
            return db_new[key]
        except ValueError:
            return db_old[key]

    def get_table(self, *args, **kwargs):
        result = db_new.get_table(*args, **kwargs)
        if len(result) == 0:
            result = db_old.get_table(*args, **kwargs)
        return result

    def get_images(self, *args, **kwargs):
        result = db_new.get_images(*args, **kwargs)
        if len(result) == 0:
            result = db_old.get_images(*args, **kwargs)
        return result

    def retrieve(self, *args, **kwargs):
        try:
            db_new.reg.retrieve(*args, **kwargs)
        except DatumNotFound:
            db_old.reg.retrieve(*args, **kwargs)


db = Broker_New.named('hxn')
