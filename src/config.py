import environ


@environ.config(prefix="")
class AppConfig:
    @environ.config(prefix="MODEL")
    class Model:
        ckpt_path = environ.var(default="src/fegan/ckpt/SC-FEGAN.ckpt")
        input_size = environ.var(default=512, converter=int)
        batch_size = environ.var(default=1, converter=int)

    model = environ.group(Model)


CONFIG: AppConfig = AppConfig.from_environ()
