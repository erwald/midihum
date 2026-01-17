# CLAUDE.md

Guidelines for working on this codebase.

## Dependencies

Follow these best practices for managing dependencies in `requirements.txt`:

1. **Only add dependencies when truly necessary** - Before adding a new package, consider if the functionality can be achieved with existing dependencies or the standard library.

2. **Remove unused dependencies** - Periodically audit dependencies and remove any that are no longer used.

3. **Don't add transitive dependencies** - If package A already depends on package B, don't add B to requirements.txt unless it's directly imported in the codebase.

4. **Search all file types** - When auditing dependencies, search both `.py` files and `.ipynb` notebooks for imports.

5. **Pin versions** - Use version constraints (e.g., `~= 1.5.0`) to ensure reproducible builds while allowing patch updates.

## Code Style

- Use lowercase for comments
- Prefix log messages with the module name (e.g., `click.echo(f"midi_scraper ...")`)
- Use `tqdm.write()` for output within loops that have progress bars

## Testing

- Tests are in `test_*.py` files
- Run tests with: `python -m pytest`
- Extract pure functions where possible to enable unit testing
