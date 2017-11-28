#ifndef GStrategy_h
#define GStrategy_h
#include <cassert>
#include "GOrder.h"
#include "GOrderHandler.h"

gpu_ginkgo{
    /*
    Here we implement the passive strategy for K7 of a gpu version.
    */
    template<const int ORDER_LIM = 200, const int LEVEL_LIM = 200, const int LEVEL_ORDER_LIM = 10>
    struct Strategy{
        gpu_ginkgo::OrderHandler<ORDER_LIM, LEVEL_LIM, LEVEL_ORDER_LIM> buy_oh;
        gpu_ginkgo::OrderHandler<ORDER_LIM, LEVEL_LIM, LEVEL_ORDER_LIM> sell_oh;
        int pos, ppos;
        int total_pnl;
        int ask_bbo;
        int bid_bbo;
        bool book_update;
    public:
        __device__ Strategy(): pos(0), ppos(0), total_pnl(0), book_update(false) {}
        
        __device__ bool getBbos(){
            return true;
        }
        
        __device__ void updateOrders(const int&ap, const i__dnt&bp, const int&t, const int ltcy, int *bz_info){
            /* Need to loop through limited_orders and canceled orders */
            sell_oh.updatePendingOrders(t, ltcy, ask_bbo, ask_bbo + 4, bz_info);
            buy_oh.updatePendingOrders(t, ltcy, bid_bbo-4, bid_bbo, bz_info);
            sell_oh.updateLimitedOrders(t+ltcy, pos, total_pnl, ask_bbo, ask_bbo + 4,
            true, ap, bp, bz_info);
            buy_oh.updateLimitedOrders(t+ltcy, pos, total_pnl, bid_bbo - 4, bid_bbo,
            false, ap, bp, bz_info);
            sell_oh.updateCancelingOrders(t, ppos, pos, total_pnl, true,
            book_update, ap, bp, bz_info);
            buy_oh.updateCancelingOrders(t, ppos, pos, total_pnl, false,
            book_update, ap, bp, bz_info);
        }
        
        __device__ void cancelPendingOrders(){
            /* Need to loop through pending order list*/
        }
        
        __device__ void sendNewOrders(const int&t, const int&ltcy){
            /* Need to loop the good range given by BBOs */
            int qty_max;
            if(ppos > 0){
                qty_max = ppos + LEVEL_ORDER_LIM;
                for(int i=0;i<5 && qty_max;++i){
                    ppos -= sell_oh.sendNewOrders(ask_bbo+i, t+ltcy, qty_max);
                }
                qty_max = LEVEL_ORDER_LIM - ppos;
                for(int i=0;i<5 && qty_max;++i){
                    ppos += sell_oh.sendNewOrders(bid_bbo-i, t+ltcy, qty_max);
                }
            }
            else{
                qty_max = LEVEL_ORDER_LIM - ppos;
                for(int i=0;i<5 && qty_max;++i){
                    ppos += sell_oh.sendNewOrders(bid_bbo-i, t+ltcy, qty_max);
                }
                qty_max = ppos + LEVEL_ORDER_LIM;
                for(int i=0;i<5 && qty_max;++i){
                    ppos -= sell_oh.sendNewOrders(ask_bbo+i, t+ltcy, qty_max);
                }
            }
        }
    };
}

#endif