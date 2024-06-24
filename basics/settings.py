import os
import dotenv

dotenv.load_dotenv()


RUST_BACKTRACE = 1


vars_to_set = {
    "RUST_BACKTRACE": RUST_BACKTRACE
}

os.environ.update(vars_to_set)
