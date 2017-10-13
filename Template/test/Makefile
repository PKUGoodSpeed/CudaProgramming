MAKEDIR = ./Makefiles
MAKEFILE = Makefile
ODIR = ./objs
CUODIR = ./cu_objs

option:
	make -f $(MAKEDIR)/$(MAKEFILE).$(option) app=app_$(option)

.PHONY: clean
clean:
	rm -f $(ODIR)/*.o $(CUODIR)/*.o
