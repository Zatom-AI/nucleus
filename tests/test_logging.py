import logging

import pytest

from nucleus.utils.logging import install_zatom_logger_name_aliases


@pytest.fixture(autouse=True)
def restore_log_record_factory():
    original_factory = logging.getLogRecordFactory()
    yield
    logging.setLogRecordFactory(original_factory)


@pytest.mark.parametrize("legacy_root", ["rf3", "rfd3", "rfd3na"])
def test_zatom_logger_name_aliases_replace_legacy_package_prefix(legacy_root):
    install_zatom_logger_name_aliases("zatom2")

    logger = logging.getLogger(f"{legacy_root}.inference.input_parsing")
    record = logger.makeRecord(logger.name, logging.INFO, __file__, 1, "msg", (), None)

    assert logger.name == f"{legacy_root}.inference.input_parsing"
    assert record.name == "zatom2.inference.input_parsing"


def test_zatom_logger_name_aliases_respect_package_boundaries():
    install_zatom_logger_name_aliases("zatom2")

    logger = logging.getLogger("rfd30.engine")
    record = logger.makeRecord(logger.name, logging.INFO, __file__, 1, "msg", (), None)

    assert record.name == "rfd30.engine"


def test_zatom_logger_name_aliases_are_idempotent():
    install_zatom_logger_name_aliases("zatom2")
    install_zatom_logger_name_aliases("zatom2")

    logger = logging.getLogger("rfd3na.utils.inference")
    record = logger.makeRecord(logger.name, logging.INFO, __file__, 1, "msg", (), None)

    assert record.name == "zatom2.utils.inference"


def test_zatom_logger_name_aliases_reject_unknown_package():
    with pytest.raises(ValueError, match="Unsupported Zatom package"):
        install_zatom_logger_name_aliases("unknown_package")
