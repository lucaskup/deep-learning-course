.PHONY: problems clean-problems exams clean-exams

# Compile every problem-set .tex file that contains \documentclass.
# Requires latexmk and a LuaTeX installation (e.g. texlive-full).
# Must be run from the repo root so that \input{problems/style/…} resolves.
problems:
	@mkdir -p build
	@FAILED=0; COMPILED=0; \
	for texfile in $$(grep -rl '\\documentclass' problems/ --include='*.tex'); do \
		echo "--- Compiling: $$texfile ---"; \
		if latexmk -lualatex -cd- "$$texfile"; then \
			pdffile="build/$$(basename "$${texfile%.tex}").pdf"; \
			if [ -f "$$pdffile" ]; then \
				echo "OK: $$pdffile"; \
				COMPILED=$$((COMPILED + 1)); \
			else \
				echo "FAIL: PDF not produced at expected path $$pdffile"; \
				FAILED=$$((FAILED + 1)); \
			fi; \
		else \
			echo "FAIL: latexmk returned non-zero for $$texfile"; \
			FAILED=$$((FAILED + 1)); \
		fi; \
	done; \
	echo ""; \
	echo "Results: $$COMPILED compiled successfully, $$FAILED failed"; \
	test $$FAILED -eq 0

clean-problems:
	latexmk -lualatex -cd- -C $$(grep -rl '\\documentclass' problems/ --include='*.tex')
	@rm -rf build/

# Compile every exam .tex file (provas and gabaritos). The CI pipeline
# excludes _gabarito.tex from the public deploy, but locally we build both.
exams:
	@mkdir -p build
	@FAILED=0; COMPILED=0; \
	for texfile in $$(grep -rl '\\documentclass' exams/ --include='*.tex'); do \
		echo "--- Compiling: $$texfile ---"; \
		if latexmk -lualatex -cd- "$$texfile"; then \
			pdffile="build/$$(basename "$${texfile%.tex}").pdf"; \
			if [ -f "$$pdffile" ]; then \
				echo "OK: $$pdffile"; \
				COMPILED=$$((COMPILED + 1)); \
			else \
				echo "FAIL: PDF not produced at expected path $$pdffile"; \
				FAILED=$$((FAILED + 1)); \
			fi; \
		else \
			echo "FAIL: latexmk returned non-zero for $$texfile"; \
			FAILED=$$((FAILED + 1)); \
		fi; \
	done; \
	echo ""; \
	echo "Results: $$COMPILED compiled successfully, $$FAILED failed"; \
	test $$FAILED -eq 0

clean-exams:
	latexmk -lualatex -cd- -C $$(grep -rl '\\documentclass' exams/ --include='*.tex')
