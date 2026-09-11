"""Exact-hash developer-venv patch, dry-run by default, reversible and fail-closed."""
import argparse
import hashlib
import os
from pathlib import Path

from ledger_x.paths import GATE_REPORT, PROJECT_ROOT

DEV = Path(os.environ.get(
    "LEDGERX_VLLM_RUNNER",
    "runtime/vllm-dev/site-packages/vllm/v1/worker/gpu_model_runner.py",
))
if not DEV.is_absolute():
    DEV = PROJECT_ROOT / DEV
ORIGINAL_ENV = os.environ.get("LEDGERX_ORIGINAL_VLLM_RUNNER")
ORIGINAL = Path(ORIGINAL_ENV) if ORIGINAL_ENV else None
PROD_ENV = os.environ.get("LEDGERX_PROD_VLLM_RUNNER")
PROD = Path(PROD_ENV) if PROD_ENV else None
ORIGINAL_HASH = "6250ca0670c4593aa2f7c4567c9c71d6cb3d8182624a985d4d4e33d5bf054666"
OLD_CALL = '''        elif spec_config.method == "custom_class":
            assert isinstance(sampled_token_ids, list)
            draft_token_ids = cast(Any, self.drafter).propose(
                sampled_token_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
                slot_mappings=slot_mappings,
            )'''
NEW_CALL = OLD_CALL.replace('            assert isinstance',
    '            from ledger_x.specdec.bridge import contexts\n            assert isinstance').replace(
    '                slot_mappings=slot_mappings,',
    '                slot_mappings=slot_mappings,\n                ledger_x_context=contexts(self, scheduler_output, hidden_states),')
OLD_FINISH = '        # Remove finished requests from the cached states.\n'
NEW_FINISH = ('        from ledger_x.specdec.bridge import finish_requests\n'
              '        finish_requests(self, scheduler_output.finished_req_ids)\n' + OLD_FINISH)


def patched(text):
    if hashlib.sha256(text.encode()).hexdigest() != ORIGINAL_HASH:
        raise RuntimeError("Unexpected runner hash; re-inspect instead of guessing offsets")
    if text.count(OLD_CALL) != 1 or text.count(OLD_FINISH) != 1:
        raise RuntimeError("Patch anchors are not unique")
    return text.replace(OLD_CALL, NEW_CALL).replace(OLD_FINISH, NEW_FINISH)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--restore", action="store_true")
    p.add_argument("--restore-original", action="store_true",
                   help="Restore DEV from LEDGERX_ORIGINAL_VLLM_RUNNER after exact hash validation")
    args = p.parse_args()
    if sum(bool(item) for item in (args.apply, args.restore, args.restore_original)) > 1:
        p.error("Choose exactly one mutating operation")
    if args.apply or args.restore or args.restore_original:
        from ledger_x.specdec.gate import dev_pid
        if dev_pid() is not None:
            raise RuntimeError("Stop the validated development instance before modifying its runner")
    if args.apply:
        import json
        gate = GATE_REPORT
        if not gate.exists() or not json.loads(gate.read_text()).get("diagnostic_passed"):
            raise RuntimeError("Diagnostic custom-path gate has not passed; bridge installation is blocked")
    if DEV.is_symlink() or DEV.resolve() != DEV or DEV.stat().st_nlink != 1:
        raise RuntimeError("Development file must be independent, non-symlink, single link")
    if PROD and PROD.exists() and DEV.stat().st_ino == PROD.stat().st_ino and DEV.stat().st_dev == PROD.stat().st_dev:
        raise RuntimeError("Refusing shared production inode")
    backup = DEV.with_suffix(".py.ledger_x-original")
    current = DEV.read_text()
    if args.restore_original:
        if ORIGINAL is None:
            raise RuntimeError("LEDGERX_ORIGINAL_VLLM_RUNNER is required for --restore-original")
        original = ORIGINAL.read_text()
        if hashlib.sha256(original.encode()).hexdigest() != ORIGINAL_HASH:
            raise RuntimeError("Original runner backup hash mismatch")
        compile(original, str(DEV), "exec")
        DEV.write_text(original)
        print("Restored exact original development runner from configured backup")
        return
    if args.restore:
        original = backup.read_text()
        if current != patched(original):
            raise RuntimeError("Runner changed after patch; refusing to overwrite")
        DEV.write_text(original)
        print("Restored exact original development runner; backup retained")
    else:
        source = backup.read_text() if backup.exists() else current
        replacement = patched(source)
        compile(replacement, str(DEV), "exec")
        if args.apply:
            if backup.exists() and hashlib.sha256(backup.read_text().encode()).hexdigest() != ORIGINAL_HASH:
                raise RuntimeError("Unexpected existing backup")
            if not backup.exists():
                if hashlib.sha256(current.encode()).hexdigest() != ORIGINAL_HASH:
                    raise RuntimeError("Current runner is not original; restore it before applying")
                backup.write_text(current)
            DEV.write_text(replacement)
            print("Applied development-only bridge; original backup retained")
        else:
            print("Dry-run passed: exact hash, unique anchors, independent inode, valid Python")


if __name__ == "__main__":
    main()
