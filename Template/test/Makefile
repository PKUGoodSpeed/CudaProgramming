MAKEDIR = ./Makefiles
MAKEFILE = Makefile
ODIR = ./objs
CUODIR = ./cu_objs

option:
	mkdir $(ODIR)
	mkdir $(CUODIR)
	make -f $(MAKEDIR)/$(MAKEFILE).$(option) app=app_$(option)

.PHONY: clean
clean:
	rm -rf $(ODIR) $(CUODIR)
