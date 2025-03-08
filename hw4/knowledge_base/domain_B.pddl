;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Op-block-L world
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Semantics
;;; count-1 = there is one block (one cell) at (1, 1)
;;; count-2 = there are two blocks (two cells) at (1, 1) and (1, 3)
;;; create-1 = add one block at (1, 3) if there is only one block
;;; delete-1 = delete one block if there are two blocks at (1, 1) and (1, 3)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain BLOCKS)
  (:requirements :strips :typing)
  (:types block)
  (:predicates (count-1 ?x)
	       (count-2 ?x)
	       )

  ;; 
  (:action create-1
     :parameters (?x - block)
     :precondition (count-1 ?x)  ;; only create if there's one block
     :effect (and 
         (not (count-1 ?x)) ;; remove one block state
         (count-2 ?x)  ;; add second block
     )
  )

  ;; Delete the second block, going back to only one block
  (:action delete-1
     :parameters (?x - block)
     :precondition (count-2 ?x)  ;; only delete if there are two blocks
     :effect (and 
         (not (count-2 ?x))  ;; remove two blocks state
         (count-1 ?x)  ;; back to one block
     )
  )

)
